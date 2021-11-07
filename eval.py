from torch import nn
from metric import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

"""
1. Validating accuracy of trained model in validation dataset and then output accuracy metric of network
2. generating csv result file
"""

def validate(validation_loader, device, model, criterion):
    model = model.to(device) # model --> GPU
    model = model.eval() # set eval mode
    with torch.no_grad():# network does not update gradient during evaluation
        val_top1 = AverageMeter()
        validate_loader = tqdm(validation_loader)
        validate_loss = 0
        for i, data in enumerate(validate_loader):
            inputs, labels = data[0].to(device), data[1].to(device) # data, label --> GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0) # batch_size=32
            val_top1.update(prec1.item(), n)
            validate_loss += loss.item()
            postfix = {'validation_loss': '%.6f' % (validate_loss / (i+1)), 'validation_acc': '%.6f'%  val_top1.avg}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return val_acc

# generate csv file to meet the requirment of dog&cat competition,
# and this function is inference program, too
def submission(csv_path, test_loader, device, model):
    result_list = []
    model = model.to(device)
    test_loader = tqdm(test_loader)
    with torch.no_grad(): # network does not update gradient during evaluation
        for i, data in enumerate(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            softmax_func = nn.Softmax(dim=1) # dim=1 means the sum of rows is 1
            soft_output = softmax_func(outputs) # soft_output is become two probability value
            predicted = soft_output[:, 1] # the probability of dog
            for i in range(len(predicted)):
                result_list.append({
                    'id': labels[i].item(),
                    'label': predicted[i].item()
                })

    # convert list to dataframe, and then generate csv format file
    columns = result_list[0].keys()
    result_list = {col: [anno[col] for anno in result_list] for col in columns}
    result_df = pd.DataFrame(result_list)
    result_df = result_df.sort_values("id")
    result_df.to_csv(csv_path, index=None)

















