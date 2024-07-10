#include <onnxruntime_cxx_api.h>
#include "../lib/tinywav/myk_tiny.h"
#include <iostream>
#include <vector>

int
main()
{
    const char* inputPath  = "/Users/thomaspower/Developer/Koala/LLVC_Test/"
                             "test_audio/174-50561-0000.wav";
    const char* outputPath = "/Users/thomaspower/Developer/Koala/LLVC_Test/"
                             "output_audio/outputsample.wav";
    const char* modelPath  = "/Users/thomaspower/Developer/Koala/LLVC_Test/"
                             "onnx_models/llvc_model.onnx";

    try
    {
        // Load audio
        int inputSampleRate, inputChannels;
        std::vector<float> audio = myk_tiny::loadWav(inputPath);
        std::vector<float> outSignal(audio.size(), 0.0f);
        if (audio.empty())
        {
            std::cerr << "Failed to load audio or audio is empty." << std::endl;
            return 1;
        }
        std::cout << "Loaded audio. Sample count: " << audio.size()
                  << ", Sample rate: " << inputSampleRate
                  << ", Channels: " << inputChannels << std::endl;

        // Set up ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "llvc_test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        Ort::Session session(env, modelPath, session_options);

        // Prepare input tensors
        Ort::MemoryInfo memory_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> input_shape = {
            1, 1, static_cast<int64_t>(audio.size())
        };
        Ort::Value input_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          audio.data(),
                                          audio.size(),
                                          input_shape.data(),
                                          input_shape.size());

        std::vector<float> enc_buf(1 * 512 * 510, 0.0f);
        std::vector<int64_t> enc_buf_shape = { 1, 512, 510 };
        Ort::Value enc_buf_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          enc_buf.data(),
                                          enc_buf.size(),
                                          enc_buf_shape.data(),
                                          enc_buf_shape.size());

        std::vector<float> dec_buf(1 * 2 * 13 * 256, 0.0f);
        std::vector<int64_t> dec_buf_shape = { 1, 2, 13, 256 };
        Ort::Value dec_buf_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          dec_buf.data(),
                                          dec_buf.size(),
                                          dec_buf_shape.data(),
                                          dec_buf_shape.size());

        std::vector<float> out_buf(1 * 512 * 4, 0.0f);
        std::vector<int64_t> out_buf_shape = { 1, 512, 4 };
        Ort::Value out_buf_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          out_buf.data(),
                                          out_buf.size(),
                                          out_buf_shape.data(),
                                          out_buf_shape.size());

        std::vector<float> convnet_pre_ctx(1 * 1 * 24, 0.0f);
        std::vector<int64_t> convnet_pre_ctx_shape = { 1, 1, 24 };
        Ort::Value convnet_pre_ctx_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          convnet_pre_ctx.data(),
                                          convnet_pre_ctx.size(),
                                          convnet_pre_ctx_shape.data(),
                                          convnet_pre_ctx_shape.size());

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
        ort_inputs.push_back(std::move(enc_buf_tensor));
        ort_inputs.push_back(std::move(dec_buf_tensor));
        ort_inputs.push_back(std::move(out_buf_tensor));
        ort_inputs.push_back(std::move(convnet_pre_ctx_tensor));

        auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                                          input_names,
                                          ort_inputs.data(),
                                          ort_inputs.size(),
                                          output_names,
                                          5);

        // Process output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape =
          output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<float> output_audio(
          output_data, output_data + output_shape[1] * output_shape[2]);

        // Save output audio
        myk_tiny::saveWav(output_audio, 1, 16000, outputPath);
        std::cout << "Saved processed audio to " << outputPath << std::endl;
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