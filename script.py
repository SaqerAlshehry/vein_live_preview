from picamera2 import Picamera2
import segmentation_models_pytorch as smp
import torch
import cv2
import numpy as np
import time

## our model

device = torch.device("cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=2
)

model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()



#camera setup
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 720)},
    controls={"FrameRate": 30}
)
picam2.configure(config)
picam2.start()

print("Live test started. Press Q to quit.")


#capture loop
while True:
    start = time.time()

    frame = picam2.capture_array()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (704, 512))
    resized = resized / 255.0

    tensor = torch.tensor(resized, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    probs = torch.softmax(output, dim=1)
    mask = probs[0,1].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    mask_big = cv2.resize(mask, (1280, 720))

    overlay = frame.copy()
    overlay[mask_big == 255] = [50,150,50]

    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    fps = 1 / (time.time() - start)
    cv2.putText(result, f"FPS: {fps:.2f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Vein Segmentation Test", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()