#-*-coding:utf-8-*-
# function: init model
import torchvision.models as models
import torch
import torch.nn as nn

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 224

    if model_name == "resnet":
        """ Resnet18
 """
        print(' -------------------->>> Resnet 51 ')
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Dropout(0.5),nn.Linear(num_ftrs, num_classes)

        # model_ft.fc = nn.Sequential(nn.Dropout(0.99),nn.Linear(num_ftrs, num_classes),nn.Softmax(dim =1 ))
        # model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),nn.LogSoftmax())#smoothing label
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    elif model_name == "alexnet":
        """ Alexnet
 """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
 """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
 """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        # input_size = 224

    elif model_name == "densenet":
        """ Densenet
 """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        """ Inception v3
 Be careful, expects (299,299) sized images and has auxiliary output
 """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

if __name__ == "__main__":
    # Initialize the model for this run
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"
    # Number of classes in the dataset
    num_classes = 2
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True
    model_ft, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # model_ft.eval()
    img = torch.rand((9, 1, 160, 160))
    output = model_ft(img)
    print('output size : ',output.size())
