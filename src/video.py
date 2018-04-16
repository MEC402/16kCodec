import cv2



def load_video(argument_filename):
    video = []
    cap = cv2.VideoCapture(argument_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    for _ in range(0, frame_count-1):
        _, frame = cap.read()
        video.append(frame)
    return video, len(video), frame_heigth, frame_width

def write_video(filename, v):
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    isColor = False
    if len(v[0].shape) == 3:
        isColor = v[0].shape[2] == 3
    out = cv2.VideoWriter(filename, fourcc, 30.0, (v[0].shape[1],v[0].shape[0]), isColor)
    for f in range(0, len(v)):
        out.write(v[f])
    out.release()
    print("Video saved to: " + filename)
    
def show_video(name, v):
    for f in v:
      cv2.imshow(name, f)
      k = cv2.waitKey(40)
      if k == 32:
            break
    cv2.destroyAllWindows()