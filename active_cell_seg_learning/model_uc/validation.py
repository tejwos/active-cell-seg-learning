import torch
import metric.loss_metric

def validation(loader, model, device = "cpu", classes = 2):
    """
    Accuracy and other Values for Evaluation
    """
    numCorrect  = 0   
    numPixels   = 0   

    accuracyValue = 0
    countValue = 0

    loseScore   = 0
    iouValue = 0


    with torch.no_grad():
        for x, y in loader:
            x = x.to(device) #this is the Input
            y = y.to(device) # this is the Targets / True / Mask of Images 
            logits = model(x)
            preds =  torch.softmax(logits, dim=1) # was: torch.sigmoid(model(x)) now: torch.softmax(model(x))
            preds =  torch.argmax(preds, dim=1)

            numCorrect  += (preds == y).sum() / 2 # creates 2 binary matrix, so / 2 for one value
            numPixels   += torch.numel(y)
            #print(logits)
            #for element_pred, element_tar in zip(logits[:], y[:]):
            loseScore   += metric.loss_metric.focal_loss_metric(pred=logits, targets=y.long(), classes=classes) 
            accuracyValue += metric.loss_metric.accuracy(y, preds)
            countValue += preds.size(dim=0)

            iouValue += metric.loss_metric.iou_mean(pred=preds, targets=y, n_classes=classes)

    data = [(loseScore / len(loader)).item(), accuracyValue/ countValue, (numCorrect).item(), 
        iouValue / len(loader),
            ]
    return data