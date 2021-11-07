from metric import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# define train function
def train(train_loader, device, model, epochs, lr, criterion, optimizer, tensorboard_path):
    model = model.to(device) # put model to GPU
    for epoch in range(epochs):
        model.train() # set train mode
        top1 = AverageMeter() # metric
        train_loader = tqdm(train_loader) # convert to tqdm type, convenient to add the output of journal
        train_loss = 0.0
        train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch+1, epochs, 'lr:', lr))
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device) # put data to GPU
            # initial 0, clear the gradient information of last batch
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('outputs:', outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # calculate topk accuracy
            acc1, acc2 = accuracy(outputs, labels, topk=(1,2))
            n = inputs.size(0) # batch_size
            # print(n)
            top1.update(acc1.item(), n)
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            train_loader.set_postfix(log=postfix)

            # tensorboard curve drawing
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Train/Loss', loss.item(), epoch)
            writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            writer.flush()
    print('Finished Training')


