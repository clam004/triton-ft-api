import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

def prepare_tensor(name, input):

  tensor = grpcclient.InferInput(
    name, 
    input.shape, 
    np_to_triton_dtype(input.dtype)
  )

  tensor.set_data_from_numpy(input)
  
  return tensor

def generate_text(
    prompt,
    ENDPOINT_URL,
    max_gen_len = 32,
    top_k = 100, 
    top_p = 0.99, 
    num_return_sequences = 1,
    temperature = 0.9,
    repetition_penalty = 1.0,
    mystop = [
        '<',
        '[human]',
        '\n',
        '[',
    ],
    verbose = False,
):

    
    MODEl_NAME = "ensemble" 

    start_id = 220
    end_id = 50256

    OUTPUT_LEN = max_gen_len
    BEAM_WIDTH = 1
    TOP_K = top_k
    TOP_P = top_p
    REPETITION_PENALTY = repetition_penalty
    TEMPERATURE = temperature

    input0 = [[prompt] for _ in range(num_return_sequences)]

    bad_words_list = np.array([[""] for _ in range(num_return_sequences)], dtype=object)
    stop_words_list = np.array([mystop for _ in range(num_return_sequences)], dtype=object)
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
    runtime_top_k = (TOP_K * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    runtime_top_p = TOP_P * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    temp = TEMPERATURE * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    rep_penalty = REPETITION_PENALTY * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    random_seed = (100 * np.random.rand(input0_data.shape[0], 1)).astype(np.uint64)
    is_return_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
    beam_width = (BEAM_WIDTH * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    start_ids = start_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    end_ids = end_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)

    inputs = [
        prepare_tensor("INPUT_0", input0_data),
        prepare_tensor("INPUT_1", output0_len),
        prepare_tensor("INPUT_2", bad_words_list),
        prepare_tensor("INPUT_3", stop_words_list),
        prepare_tensor("runtime_top_k", runtime_top_k),
        prepare_tensor("runtime_top_p", runtime_top_p),
        prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
        prepare_tensor("temperature", temp),
        prepare_tensor("len_penalty", len_penalty),
        prepare_tensor("repetition_penalty", rep_penalty),
        prepare_tensor("random_seed", random_seed),
        prepare_tensor("is_return_log_probs", is_return_log_probs),
        prepare_tensor("beam_width", beam_width),
        prepare_tensor("start_id", start_ids),
        prepare_tensor("end_id", end_ids),
    ]

    client = grpcclient.InferenceServerClient(ENDPOINT_URL, verbose=False)
    result = client.infer(MODEl_NAME, inputs)
    generated_text = result.as_numpy("OUTPUT_0")
    outputs = [o.decode("utf-8") for o in generated_text]

    return outputs

print(generate_text('the first rule of robotics is','20.112.126.140:2001'))