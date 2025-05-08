import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ====== LOAD ENGINE ======
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# ====== ALLOCATE BUFFER ======
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    
    return inputs, outputs, bindings, stream

# ====== INFERENCE ======
def infer(context, bindings, inputs, outputs, stream, image):
    [host_input, device_input] = inputs[0]
    [host_output, device_output] = outputs[0]

    np.copyto(host_input, image.ravel())

    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    return host_output

# ====== PREPROCESS ======
def preprocess_image(img, input_shape=(640, 640)):
    img_resized = cv2.resize(img, input_shape)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_trans = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(img_trans, axis=0)

# ====== MAIN ======
if __name__ == "__main__":
    engine_path = "models/train3.engine"  # path ke TensorRT engine
    input_shape = (640, 640)

    print("[INFO] Loading engine...")
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_input = preprocess_image(frame, input_shape)
        t0 = time.time()
        output = infer(context, bindings, inputs, outputs, stream, img_input)
        print(f"Inference time: {(time.time() - t0)*1000:.2f} ms")

        # TODO: Decode output and draw boxes (bisa pakai NMS + confidence filter)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
