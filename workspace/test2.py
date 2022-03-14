import cv2
import time
import db_infer
import torch
import torch.nn.functional as F
import numpy as np
from utils import nms, uexp
import torchvision.transforms as T

# db_infer.build('/data-nbd/lxh/DeepBlueAI_Face3/detector/dbface.onnx', 'dbface.trt', 16)
dbface = db_infer.Infer('dbface.trt')

mean = [0.408, 0.447, 0.47]
std = [0.289, 0.274, 0.278]
trans = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
    T.Lambda(lambda x : x.unsqueeze(dim=0))
])

def pad(image, stride=32):
    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True 

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        # newImage = cp.zeros((stdh, stdw, 3), cp.uint8)
        newImage = np.zeros((stdh, stdw, 3), dtype=np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
        return newImage
    else:
        return image

def imread(image_or_path):
    torch_images = []
    for img_path in image_or_path:
        image = cv2.imread(img_path)
        image = cv2.resize(image, (1024, 768))

        image = pad(image)
        torch_image = trans(image)
        torch_images.append(torch_image)
    torch_image = torch.cat(torch_images, dim=0)

    return torch_image

def detect(torch_image, scale=1.0):
    # batch = torch_image.shape[0]
    btscores, bboxs, blandmarks  = dbface.inference(torch_image)
    # hm = torch.Tensor(hm)
    # hm_pool = torch.Tensor(hm_pool)
    # box = torch.Tensor(box)
    # landmark = torch.Tensor(landmark)
    # print(hm.shape, hm_pool.shape, box.shape, landmark.shape)
    # scores, indices     = ((hm == hm_pool).float() * hm).view(batch, -1).topk(1000)
    # # scores, indices     = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    # hm_height, hm_width = hm.shape[2:]

    # # print("scores.shape = ", scores.shape)
    # scores   = scores.squeeze()
    # indices  = indices.squeeze()
    # if batch == 1:
    #     scores = scores[None]
    #     indices = indices[None]
    # ys       = list(torch.true_divide(indices, hm_width).int().data.numpy())
    # xs       = list((indices % hm_width).int().data.numpy())
    # scores   = list(scores.data.numpy())
    # box      = box.cpu().squeeze().data.numpy()
    # landmark = landmark.cpu().squeeze().data.numpy()
    # # print(len(xs), xs[0])
    # if batch == 1:
    #     box = box[None]
    #     landmark = landmark[None]

    # btscores, bboxs, blandmarks = [], [],[]
    # for i in range(len(xs)):
    #     tscores, boxs, landmarks = [], [],[]
    #     for cx, cy, score in zip(xs[i], ys[i], scores[i]):
    #         if score < 0.5:
    #             continue
    #         # print(cy, cx)
    #         x, y, r, b = box[i, :, cy, cx]
    #         xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * 4 * scale
    #         x5y5 = landmark[i, :, cy, cx]
    #         x5y5 = (uexp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * 4 * scale
    #         box_landmark = list(zip(x5y5[:5], x5y5[5:]))
            
    #         if((xyrb[2]-xyrb[0])*(xyrb[3]-xyrb[1]) > 400):
    #             tscores.append(score)
    #             boxs.append(xyrb)
    #             landmarks.append(box_landmark)
    #     tscores, boxs, landmarks = nms(tscores, boxs, landmarks)
    #     btscores.append(tscores)
    #     bboxs.append(boxs)
    #     blandmarks.append(landmarks)
    return btscores, bboxs, blandmarks

if __name__ == '__main__':
    img_path1 = "/data-nbd/lxh/DeepBlueAI_Face3/image.jpg"
    img_path2 = "/data-nbd/lxh/other/DBFace-master/datas/12_Group_Group_12_Group_Group_12_728.jpg"
    torch_image = imread([img_path1, img_path2])
    print(torch_image.shape)
    s, b, l = detect(torch_image)
    image = cv2.imread(img_path1)
    # image = cv2.resize(image, (1024, 768))
    for box, landmark in zip(b[0], l[0]):
        try:
            print("this: ", box[0], box[1], box[2], box[3])
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2, 16)
            for i in range(5):
                x, y = landmark[:, i]
                cv2.circle(image, (int(x), int(y)), 3, (0,0,255), -1, 16)
        except:
            print("this")
            # print("this: ", box[0], box[1], box[2], box[3])
    cv2.imwrite("res.jpg", image)
    print(len(s[0]), len(b[0]), len(l[0]))