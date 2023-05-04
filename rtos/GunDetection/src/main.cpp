#include "esp_adc_cal.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include <arduinoFFT.h>
#include <TTN_esp32.h>

#define SAMPLES 128                 // Must be a power of 2
#define SAMPLING_FREQ 8192           // Hz, must be 40000 or less due to ADC conversion time. Determines maximum frequency that can be analysed by the FFT Fmax=sampleF/2.
#define NUM_SAMPLES (SAMPLING_FREQ * 2) // Number of samples to take in 3 seconds

TTN_esp32 ttn ;
const char *appEui = "0000000000000000";
const char *appKey = "C2F2B00E573E395D7B2EEF4CD11E0C87";
const char *devEui = "70B3D57ED005A12D";

tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
constexpr int kTensorArenaSize = 67700; // the exact amount needed for this model
alignas(16) uint8_t tensor_arena[kTensorArenaSize];


long predictedLastSince = millis();

SemaphoreHandle_t binarySemaphore = xSemaphoreCreateBinary();


void predict(void *parameter);
void startUpModel();
void computeSound(void *parameter);
void sendMessage(void *parameter);

void setup() {
  Serial.begin(115200);
  Serial.println("starting");
  ttn.begin();
  ttn.join(devEui, appEui, appKey);
  uint8_t i = 0;
  while (!ttn.isJoined()) {
    Serial.print(".");
    vTaskDelay(500);
    if (i++ > 40) {
      Serial.println("Failed to join TTN");
      ttn.join(devEui, appEui, appKey);
      i = 0;
    }
  }
  Serial.println("Joined TTN");
  delay(3000);

  Serial.println("starting model");
  startUpModel();  
  Serial.println("model started");
  delay(3000);

  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_11);  
}

void loop() {
  xSemaphoreGive(binarySemaphore);

  float *samples = new float[NUM_SAMPLES];
  unsigned long timer = millis();
  unsigned int counter = 0;

  int bitDepth = 12;
  float analog_min = 0;
  float analog_max = 4095;
  float output_min = -pow(2, bitDepth-1);
  float output_max = pow(2, bitDepth-1);

  while(1) {
    delayMicroseconds(78);
    float x = map(adc1_get_raw(ADC1_CHANNEL_0), analog_min, analog_max, output_min, output_max);
    x = (x - output_min) / (output_max - output_min) * 2 -1.65;
    samples[counter] = x;

    counter++;
    if (counter == NUM_SAMPLES) {

      counter = 0;  

      xSemaphoreTake(binarySemaphore, portMAX_DELAY);
      xTaskCreate(computeSound, "computeSound", 1700 * 2, samples, 2, NULL);
      
      vTaskDelay(10);
      delayMicroseconds(10);
      Serial.print("Recoded 2 seconds of sound in: ");
      Serial.print(millis() - timer);
      Serial.print("ms\n");
      Serial.print("ReRecording\n");

      timer = millis();
    
    }
  } 


}

void startUpModel() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
      TF_LITE_REPORT_ERROR(error_reporter,
                            "Model provided is schema version %d not equal "
                            "to supported version %d.",
                            model->version(), TFLITE_SCHEMA_VERSION);
      while (1) {}
  }

  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddReshape();
  resolver.AddExpandDims();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(
  model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      while (1) {}
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("input shape: [");
  for (int i = 0; i < input->dims->size; i++) {
      Serial.print(input->dims->data[i]);
      if (i < input->dims->size - 1) {
          Serial.print(", ");
      }
  }
  Serial.println("]");

  if (input->type == kTfLiteFloat32) {
    Serial.println("input data type: float32");
  } else if (input->type == kTfLiteUInt8) {
      Serial.println("input data type: uint8");
  } else if (input->type == kTfLiteInt8) {
      Serial.println("input data type: int8");
  } else if (input->type == kTfLiteInt16) { 
    Serial.println("input data type: int16");
  } else {
      Serial.println("input data type: unknown");
  }

  Serial.print("output shape: [");
  for (int i = 0; i < output->dims->size; i++) {
      Serial.print(output->dims->data[i]);
      if (i < output->dims->size - 1) {
          Serial.print(", ");
      }
  }
  Serial.println("]");

  if (output->type == kTfLiteFloat32) {
    Serial.println("output data type: float32");
  } else if (output->type == kTfLiteUInt8) {
      Serial.println("output data type: uint8");
  } else if (output->type == kTfLiteInt8) {
      Serial.println("output data type: int8");
  } else if (output->type == kTfLiteInt16) { 
    Serial.println("output data type: int16");
  } else {
      Serial.println("output data type: unknown");
  }
}



void computeSound(void *parameter) {
  float *samples = (float*)parameter;
  input->data.f[0] = 1.0f;
  long timer = millis();

  for (int i = 0; i < SAMPLES; i++) {  
    int N = SAMPLES + SAMPLES;
    if (i == 0 || i == (SAMPLES - 1)) {
      N -= SAMPLES/2;
    }
    double *vReal = new double[N]{0};
    double *vImag = new double[N]{0};
    if (i == 0) {
      for (int j = 0; j < N; j++) {
        vReal[j] = samples[i * SAMPLES + j];
      }
    }
    if (i == (SAMPLES - 1)) {
      for (int j = 0; j < ((N - SAMPLES) / 2); j++) {
        vReal[j] = samples[i * SAMPLES - ((N - SAMPLES) / 2) + j];
      }
      for (int j = 0; j < SAMPLES; j++) {
        vReal[j + ((N - SAMPLES) / 2)] = samples[i * SAMPLES + j];
      }
    }
    else {
     for (int j = 0; j < ((N - SAMPLES) / 2); j++) {
        vReal[j] = samples[i * SAMPLES - ((N - SAMPLES) / 2) + j];
      }
      for (int j = 0; j < SAMPLES + ((N - SAMPLES) / 2) ; j++) {
        vReal[j + ((N - SAMPLES) / 2)] = samples[i * SAMPLES + j];
      }
    }
    arduinoFFT FFT = arduinoFFT(vReal, vImag, SAMPLES, SAMPLING_FREQ);
    FFT.DCRemoval();
    FFT.Windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    FFT.Compute(FFT_FORWARD);
    FFT.ComplexToMagnitude();
  
    for (int j = 0; j < SAMPLES/2; j++) {
      input->data.f[i * j+1] = std::abs(vReal[j]);
    }
    delete(vReal);
    delete(vImag);
  }

  Serial.print("Done Computing Sound in: ");
  Serial.print(millis() - timer);
  Serial.print(" ms\n");

  xTaskCreate(predict, "predict", 20000, NULL, 1, NULL); 
  vTaskDelete(NULL);

}

void predict(void *parameter) {
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.print("Invoke failed on index: ");
    Serial.println(invoke_status);
    return;
  }

  double prediction = (double)output->data.f[1];

  Serial.print("Predicted: "); 
  Serial.print(prediction);
  Serial.println();

  if (prediction > 0.5) {
    if (prediction > 1.0) {
      Serial.print("Microphone recorded improper data\n");
      xSemaphoreGive(binarySemaphore);

    }
    else {
      xTaskCreate(sendMessage, "message", 1700, NULL, 1, NULL);
    }
  } else {
    xSemaphoreGive(binarySemaphore);
  }
  
  Serial.print("PredictedLastSince: ");
  Serial.print(millis() - predictedLastSince);
  Serial.print(" ms\n");
  predictedLastSince = millis();
  
  vTaskDelete(NULL);
}


void sendMessage(void *parameter) {

  uint8_t *data = new uint8_t[4] {0x48, 0x49, 0x54, 0x21};
  ttn.sendBytes(data, 4);
  Serial.print("Gunshot Detected, Message Sent!\n");
  xSemaphoreGive(binarySemaphore);

  vTaskDelete(NULL);
}



