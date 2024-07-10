#include <portaudio.h>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

// Takes in a ref to the session, and does inference on the input block
void
processBlock(Ort::Session& session,
             std::vector<float>& block,
             std::unique_ptr<Ort::Value>& enc_buf_tensor,
             std::unique_ptr<Ort::Value>& dec_buf_tensor,
             std::unique_ptr<Ort::Value>& out_buf_tensor,
             std::unique_ptr<Ort::Value>& convnet_pre_ctx_tensor)
{
    // Prepare input tensors
    Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = { 1,
                                         1,
                                         static_cast<int64_t>(block.size()) };
    Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(memory_info,
                                      block.data(),
                                      block.size(),
                                      input_shape.data(),
                                      input_shape.size());

    // Prepare input and output names
    const char* input_names[] = {
        "input", "enc_buf", "dec_buf", "out_buf", "convnet_pre_ctx"
    };
    const char* output_names[] = { "output",
                                   "new_enc_buf",
                                   "new_dec_buf",
                                   "new_out_buf",
                                   "new_convnet_pre_ctx" };

    // Run inference
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(*enc_buf_tensor));
    ort_inputs.push_back(std::move(*dec_buf_tensor));
    ort_inputs.push_back(std::move(*out_buf_tensor));
    ort_inputs.push_back(std::move(*convnet_pre_ctx_tensor));

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                                      input_names,
                                      ort_inputs.data(),
                                      ort_inputs.size(),
                                      output_names,
                                      5);

    // Process output writing to the block
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> output_shape =
      output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<float> output_audio(
      output_data, output_data + output_shape[1] * output_shape[2]);

    // copy output_audio to block
    std::copy(output_audio.begin(), output_audio.end(), block.begin());

    // Update state tensors with new values
    enc_buf_tensor = std::make_unique<Ort::Value>(std::move(output_tensors[1]));
    dec_buf_tensor = std::make_unique<Ort::Value>(std::move(output_tensors[2]));
    out_buf_tensor = std::make_unique<Ort::Value>(std::move(output_tensors[3]));
    convnet_pre_ctx_tensor =
      std::make_unique<Ort::Value>(std::move(output_tensors[4]));
}

const int BLOCK_SIZE = 512;

struct AudioData
{
    std::vector<float> buffer;
    Ort::Session* session;
    std::unique_ptr<Ort::Value>* enc_buf_tensor;
    std::unique_ptr<Ort::Value>* dec_buf_tensor;
    std::unique_ptr<Ort::Value>* out_buf_tensor;
    std::unique_ptr<Ort::Value>* convnet_pre_ctx_tensor;
};

static int
paCallback(const void* inputBuffer,
           void* outputBuffer,
           unsigned long framesPerBuffer,
           const PaStreamCallbackTimeInfo* timeInfo,
           PaStreamCallbackFlags statusFlags,
           void* userData)
{
    AudioData* data = (AudioData*)userData;
    const float* in = (const float*)inputBuffer;
    float* out      = (float*)outputBuffer;

    std::vector<float> block(in, in + framesPerBuffer);
    processBlock(*data->session,
                 block,
                 *data->enc_buf_tensor,
                 *data->dec_buf_tensor,
                 *data->out_buf_tensor,
                 *data->convnet_pre_ctx_tensor);

    std::copy(block.begin(), block.end(), out);

    return paContinue;
}

int
main()
{
    const char* modelPath = "/Users/thomaspower/Developer/Koala/LLVC_Test/"
                            "onnx_models/llvc_model.onnx";

    try
    {
        // Set up ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "llvc_realtime");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        Ort::Session session(env, modelPath, session_options);

        // Initialize state tensors
        Ort::MemoryInfo memory_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<float> enc_buf(1 * 512 * 510, 0.0f);
        std::vector<int64_t> enc_buf_shape = { 1, 512, 510 };
        auto enc_buf_tensor                = std::make_unique<Ort::Value>(
          Ort::Value::CreateTensor<float>(memory_info,
                                          enc_buf.data(),
                                          enc_buf.size(),
                                          enc_buf_shape.data(),
                                          enc_buf_shape.size()));

        std::vector<float> dec_buf(1 * 2 * 13 * 256, 0.0f);
        std::vector<int64_t> dec_buf_shape = { 1, 2, 13, 256 };
        auto dec_buf_tensor                = std::make_unique<Ort::Value>(
          Ort::Value::CreateTensor<float>(memory_info,
                                          dec_buf.data(),
                                          dec_buf.size(),
                                          dec_buf_shape.data(),
                                          dec_buf_shape.size()));

        std::vector<float> out_buf(1 * 512 * 4, 0.0f);
        std::vector<int64_t> out_buf_shape = { 1, 512, 4 };
        auto out_buf_tensor                = std::make_unique<Ort::Value>(
          Ort::Value::CreateTensor<float>(memory_info,
                                          out_buf.data(),
                                          out_buf.size(),
                                          out_buf_shape.data(),
                                          out_buf_shape.size()));

        std::vector<float> convnet_pre_ctx(1 * 1 * 24, 0.0f);
        std::vector<int64_t> convnet_pre_ctx_shape = { 1, 1, 24 };
        auto convnet_pre_ctx_tensor = std::make_unique<Ort::Value>(
          Ort::Value::CreateTensor<float>(memory_info,
                                          convnet_pre_ctx.data(),
                                          convnet_pre_ctx.size(),
                                          convnet_pre_ctx_shape.data(),
                                          convnet_pre_ctx_shape.size()));

        // Initialize PortAudio
        PaError err = Pa_Initialize();
        if (err != paNoError)
        {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err)
                      << std::endl;
            return 1;
        }

        AudioData data = { std::vector<float>(BLOCK_SIZE),
                           &session,
                           &enc_buf_tensor,
                           &dec_buf_tensor,
                           &out_buf_tensor,
                           &convnet_pre_ctx_tensor };
        PaStream* stream;
        err = Pa_OpenDefaultStream(&stream,
                                   1,          // Input channels
                                   1,          // Output channels
                                   paFloat32,  // Sample format
                                   16000,      // Sample rate
                                   BLOCK_SIZE, // Frames per buffer
                                   paCallback, // Callback function
                                   &data);     // User data
        if (err != paNoError)
        {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err)
                      << std::endl;
            return 1;
        }

        err = Pa_StartStream(stream);
        if (err != paNoError)
        {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err)
                      << std::endl;
            return 1;
        }

        std::cout << "Press Enter to stop..." << std::endl;
        std::cin.get();

        err = Pa_StopStream(stream);
        if (err != paNoError)
        {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err)
                      << std::endl;
            return 1;
        }

        err = Pa_CloseStream(stream);
        if (err != paNoError)
        {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err)
                      << std::endl;
            return 1;
        }

        Pa_Terminate();
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}