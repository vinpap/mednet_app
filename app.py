import streamlit as st
import pickle
import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from PIL import Image



######################### CODE PYTORCH #####################
# À cause de la façon dont Streamlit fonctionne, il est beaucoup plus simple de mettre tout le code dans un seul fichier


if torch.cuda.is_available():     # Make sure GPU is available
    print("Computing with GPU")
    dev = torch.device("cuda:0")
    kwar = {'num_workers': 8, 'pin_memory': True}
    cpu = torch.device("cpu")
else:
    print("Warning: CUDA not found, CPU only.")
    dev = torch.device("cpu")
    kwar = {}
    cpu = torch.device("cpu")

class MedNet(nn.Module):
    def __init__(self,xDim,yDim,numC, num_convs=(5, 10), conv_sizes=(7, 7), fc_sizes=(400, 80), add_dropout=False): # Pass image dimensions and number of labels when initializing a model   
        super(MedNet,self).__init__()  # Extends the basic nn.Module to the MedNet class
        # The parameters here define the architecture of the convolutional portion of the CNN. Each image pixel
        # has numConvs convolutions applied to it, and convSize is the number of surrounding pixels included
        # in each convolution. Lastly, the numNodesToFC formula calculates the final, remaining nodes at the last
        # level of convolutions so that this can be "flattened" and fed into the fully connected layers subsequently.
        # Each convolution makes the image a little smaller (convolutions do not, by default, "hang over" the edges
        # of the image), and this makes the effective image dimension decreases.
        
        self.add_dropout = add_dropout
        
        numConvs1 = num_convs[0]
        convSize1 = conv_sizes[0]
        numConvs2 = num_convs[1]
        convSize2 = conv_sizes[1]
        numNodesToFC = numConvs2*(xDim-(convSize1-1)-(convSize2-1))*(yDim-(convSize1-1)-(convSize2-1))

        # nn.Conv2d(channels in, channels out, convolution height/width)
        # 1 channel -- grayscale -- feeds into the first convolution. The same number output from one layer must be
        # fed into the next. These variables actually store the weights between layers for the model.
        
        self.cnv1 = nn.Conv2d(1, numConvs1, convSize1)
        self.cnv2 = nn.Conv2d(numConvs1, numConvs2, convSize2)

        # These parameters define the number of output nodes of each fully connected layer.
        # Each layer must output the same number of nodes as the next layer begins with.
        # The final layer must have output nodes equal to the number of labels used.
        
        fcSize1 = fc_sizes[0]
        fcSize2 = fc_sizes[1]
        
        # nn.Linear(nodes in, nodes out)
        # Stores the weights between the fully connected layers
        
        self.ful1 = nn.Linear(numNodesToFC,fcSize1)
        if self.add_dropout: self.drop1 = nn.Dropout(0.5)
        self.ful2 = nn.Linear(fcSize1, fcSize2)
        if self.add_dropout: self.drop2 = nn.Dropout(0.5)
        self.ful3 = nn.Linear(fcSize2,numC)
        
    def forward(self,x):
        # This defines the steps used in the computation of output from input.
        # It makes uses of the weights defined in the __init__ method.
        # Each assignment of x here is the result of feeding the input up through one layer.
        # Here we use the activation function elu, which is a smoother version of the popular relu function.
        
        x = F.elu(self.cnv1(x)) # Feed through first convolutional layer, then apply activation
        x = F.elu(self.cnv2(x)) # Feed through second convolutional layer, apply activation
        x = x.view(-1,self.num_flat_features(x)) # Flatten convolutional layer into fully connected layer
        x = F.elu(self.ful1(x)) # Feed through first fully connected layer, apply activation
        if self.add_dropout: x = self.drop1(x)
        x = F.elu(self.ful2(x)) # Feed through second FC layer, apply output
        if self.add_dropout: x = self.drop2(x)
        x = self.ful3(x)        # Final FC layer to output. No activation, because it's used to calculate loss
        return x

    def num_flat_features(self, x):  # Count the individual nodes in a layer
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def scaleImage(x):          # Pass a PIL image, return a tensor
    x = x.convert("L")
    x = x.resize((64, 64))
    toTensor = tv.transforms.ToTensor()
    y = toTensor(x)
    if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min())/(y.max() - y.min()) 
    z = y - y.mean()        # Subtract the mean value of the image
    return z

def predict(images): 
    x = torch.stack([scaleImage(Image.open(i)) for i in images]).to(dev)

    results = model(x)
    class_indices = torch.max(results, 1)[1]
    return class_indices



model = pickle.load(open("best_model.pkl", "rb"))




################## CODE STREAMLIT ###################

classes = {
    0: "ChestCT",
    1: "CXR",
    2: "BreastMRI",
    3: "Hand",
    4: "HeadCT",
    5: "AbdomenCT"
}


def submit_images(images):

    st.session_state["images"] = images

    try:
        result = predict(images)
        st.session_state["results"] = result
        st.session_state["display_results"] = True
    except:
        st.write("Une erreur est survenue lors du traitement des images !")

def go_back_to_form():
    st.session_state["display_results"] = False


if "display_results" in st.session_state and st.session_state["display_results"] == True: 
    st.markdown("<h1 style='text-align: center; color: red; font-size:40px'>Classes prédites</h1>", unsafe_allow_html=True)

    for index in range(len(st.session_state["images"])):
        col1, col2 = st.columns(2)
        col1.image(st.session_state["images"][index], channels="BGR", width=200)
        class_name = classes[st.session_state["results"][index].item()]
        col2.markdown(f"<div style='text-align: center; vertical-align: middle; font-size: 20px'>{class_name}</div>", unsafe_allow_html=True)
    st.button(
        label="Revenir à la page d'accueil",
        on_click=go_back_to_form,
        type="primary"
        )


else:
    
    st.markdown("<h1 style='text-align: center; color: red; font-size:60px'>Bienvenue sur le classificateur Mednet !</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:20px'>Choisissez une ou plusieurs images que vous souhaitez classifier :</p>", unsafe_allow_html=True)

    images = st.file_uploader(
        label="Choisir des fichiers",
        type=['png', 'jpg', 'gif', 'jpeg'],
        accept_multiple_files=True
        )

    if images is not None:

        st.image(images, channels="BGR", width=150)
        st.button(
            label=":arrow_down: Classifier",
            on_click=submit_images,
            kwargs={"images": images},
            type="primary"
            )
    else:
        st.button(
            label="Classifier",
            disabled=True,
            type="primary"
        )



