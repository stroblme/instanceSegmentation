import cv2
import torch
import torchvision
import time 

class utils:
    def init(self):
        # See if GPU is available and if yes, use it
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(10)

        # Define the standard transforms that need to be done at inference time
        self.imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
        self.preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean = self.imagenet_stats[0],
                                                                                      std  = self.imagenet_stats[1])])

    def load_model(self):
        # Load the DeepLab v3 model to system
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
        self.model.to(self.device).eval()
        # return self.model

    def grab_frame(self, cap):
        # Given a video capture object, read frames from the same and convert it to RGB
        _, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def get_pred(self, img):
        start = time.time()
        

        initT = time.time()
        print(f"initT = {initT - start}")

        input_tensor = self.preprocess(img).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        prepT = time.time()
        print(f"prepT = {prepT - initT}")

        # Make the predictions for labels across the image
        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
            output = output.argmax(0)

        predT = time.time()
        print(f"predT = {predT - prepT}")
        
        # Return the predictions
        return output.cpu().numpy()