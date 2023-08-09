import cv2
import os
from PIL import Image
import torch
import open_clip
import time
from datetime import timedelta


def puttext(image, text):
    """
    Put text using OpenCV
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (100, 50)
    fontScale = 2
    color = (255, 0, 0)
    thickness = 2
    return cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)


def run(source, pretrained, classes, device, show_video, save):
    """
    Infer OpenClip model on video
    """
    model, _, preprocess = open_clip.create_model_and_transforms(*pretrained)
    tokenizer = open_clip.get_tokenizer(pretrained[0])
    text = tokenizer(classes).to(device)
    model.to(device)

    cap = cv2.VideoCapture(source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(os.path.basename(source), 
                             cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame2show = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = preprocess(frame).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(frame)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, labels = torch.max(text_probs, dim=1)
            frame2show = puttext(frame2show, classes[labels[0].item()])
            if show_video:
                cv2.imshow("Frame", frame2show)
            if save:
                result.write(frame2show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    result.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    source = "test_samples/chay_ngoai_xuong_2.mp4"
    classes = ["fire or smoke", "none"]
    pretrained = ('ViT-L-14', 'commonpool_xl_s13b_b90k') # open_clip.list_pretrained()
    # pretrained = ('ViT-bigG-14', 'laion2b_s39b_b160k') # More than 2.5B params
    device = torch.device("cuda:1")
    show_video = False
    save = True
    start = time.time()
    run(source, pretrained, classes, device, show_video, save)
    end = time.time()
    print(f"Running time: {timedelta(seconds=round(end - start, 0))}")


