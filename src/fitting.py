from utils import AverageMeter

import time

def validate_model(val_loader, model, criterion, epoch, plot=True, use_gpu=False):
    print('='*10, 'Starting validation epoch {}'.format(epoch), '='*10) 
    model.eval()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    already_saved_images = False
    for i, (stpt, imc) in enumerate(val_loader):
        data_time.update(time.time() - end)

        # Use GPU
        if use_gpu: 
            stpt, imc = stpt.cuda(), imc.cuda()

        # Run model and record loss
        imc_recons = model(stpt.double()).cuda() # throw away class predictions
        loss = criterion(imc_recons.double(), imc.double())
        losses.update(loss.item(), stpt.size(0))

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 25 == 0:
          print('Validate: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                 i, len(val_loader), batch_time=batch_time, loss=losses))
    
    return losses.avg


def train_model(train_loader, model, criterion, optimizer, epoch, plot=True, use_gpu=False):
    print('='*10, 'Starting training epoch {}'.format(epoch), '='*10)
    model.train()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (stpt, imc) in enumerate(train_loader):
    
        # Use GPU if available
        if use_gpu:
            stpt, imc = stpt.cuda(), imc.cuda()

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        imc_recons = model(stpt.double()).cuda()
        loss = criterion(imc_recons.double(), imc.double()) 
        losses.update(loss.item(), stpt.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses)) 