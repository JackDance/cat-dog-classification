import torch

# using class to save and update accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

# define top-k accuracy
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0) # 32
    # get top-k index
    _, pred = output.topk(maxk, 1, True, True) # using top-k to get the index of the top k
    pred = pred.t() # transpose
    # eq: compare according to corresponding elements; view(1, -1): automatically converted to the shape of row 1, ;
    # expand_as(pred): shape extended to 'pred'
    # expand_as performs row-by-row replication to expand, and ensure that the columns are equal
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    # print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0) # flatten the data of first k row to one dim to get total numbers of true
        rtn.append(correct_k.mul_(100.0 / batch_size)) # mul_() tensor's multiplication, (acc num/total num)*100 to become percentage
    return rtn


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # using the device of net if not signing device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # eval mode, this operation will close dropout
            # calculate the correct predict number in a batch
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # change back to train mode
            # add the number of batch size
            n += y.shape[0]
    return acc_sum /n




