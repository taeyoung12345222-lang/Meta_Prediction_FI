import torch
import torch.nn.functional as F
from torch import nn
from torch.quantization import QuantStub, DeQuantStub

# quantised versions of ResNet8 and BasicBlock
class QBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, q=False):
        super(QBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.q = q
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.q:
            out = self.dequant(out)
            shortcut_x = self.shortcut(x)
            out += self.dequant(shortcut_x)
            out = self.quant(out)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class QResNet8(nn.Module):
    def __init__(self, num_classes=10, q=False):
        super(QResNet8, self).__init__()
        self.q = q
        block = QBasicBlock
        num_blocks = [1,1,1]
        self.num_classes = num_classes
        self.in_planes = 128

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(block,  128, num_blocks[0], stride=1, q=self.q)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2, q=self.q)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2, q=self.q)
        self.linear1 = nn.Linear(2048, num_classes) 
        self.linear2 = nn.Linear(2048, num_classes)
        self.emb = nn.Embedding(num_classes, num_classes)
        self.emb.weight = nn.Parameter(torch.eye(num_classes))
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride, q=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, q))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.q:
            x = self.quant(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out) #b*128*32*32
        out = self.layer2(out)#b*256*16*16
        out = self.layer3(out) #b*512*8*8
        self.inner = out
        out = F.avg_pool2d(out, 4)
        out = out.reshape(out.shape[0],512*2*2)
        
        self.flatten_feat = out #b*2048
        out = self.linear1(out)
        if self.q:
            out = self.dequant(out)
        return out

    def get_attentions(self):
        inner_copy = self.inner.detach().clone()#b*512*8*8
        inner_copy.requires_grad = True
        out = F.avg_pool2d(inner_copy, 4)#b*512*2*2
        out = out.view(out.size(0), -1)#b*2048
        out = self.linear1(out)#b*num_classes
        losses = out.sum(dim=0)# num_classes
        cams = []
        #import ipdb;ipdb.set_trace()
        #assert losses.shape ==self.num_classes
        for n in range(self.num_classes):
            loss = losses[n]
            self.zero_grad()
            if n<self.num_classes-1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            grads_val = inner_copy.grad
            weights = grads_val.mean(dim=(2, 3), keepdim=True)#b*512*1*1
            cams.append(F.relu((weights.detach() * self.inner).sum(dim=1)))#b*8*8
        atts = torch.stack(cams, dim=1)
        return atts