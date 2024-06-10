import torch
import copy
import os
from torchvision import transforms
from tqdm import tqdm
from hyperbox.mutator import DartsMutator

from unified_net.mobilenet3d.network import Mobile3DNet
from unified_net.mobilenet3d.augmentation import *
from unified_net.mobilenet3d.utils import *
from unified_net.mobilenet3d.ops import *
from unified_net.mobilenet3d.transform import ShapeTransform3D
from medmnist import NoduleMNIST3D



# Define data transformation pipelines
train_transform = transforms.Compose(
                [
                    ShapeTransform3D('random'),
                ]
            )

val_transform = transforms.Compose(
                [
                    ShapeTransform3D(0.5),
                ]
)

os.makedirs("data", exist_ok=True)

# Downlload and split data
train_set = NoduleMNIST3D(root='./data', split='train', download=True, transform=train_transform)
val_set = NoduleMNIST3D(root='./data', split='train', download=True, transform=val_transform)
test_set = NoduleMNIST3D(root='./data', split='train', download=True, transform=val_transform)

len_train_set = len(train_set)
len_val_set = len(train_set)
len_test_set = len(train_set)

print(f"length of train/val/test set: {len_train_set}/{len_val_set}/{len_test_set}")


# Define data loader over dataset
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)



# Define newtwork
net = Mobile3DNet(
    num_classes=1,
    in_channels=1
)

dm = DartsMutator(net)

dm.reset()

network_opt = torch.optim.SGD(net.parameters(),  lr=0.01, momentum=0.9, weight_decay=5e-4)
architecture_opt = torch.optim.Adam(dm.parameters(), lr=3e-4, weight_decay=1e-3)




# Train utility functions
def unrolled_backward(model, mutator, criterion, w_opt, a_opt, train_x, train_y, val_x, val_y):
    backup_params = copy.deepcopy(tuple(model.parameters()))

    # do virtual step on training data
    # _compute_virtual_model
    lr = w_opt.param_groups[0]["lr"]
    momentum = w_opt.param_groups[0]["momentum"]
    weight_decay = w_opt.param_groups[0]["weight_decay"]
    mutator.reset()
    output = model(train_x)
    w_loss = criterion(output, train_y)
    gradients = torch.autograd.grad(w_loss, model.parameters())
    with torch.no_grad():
        for w, g in zip(model.parameters(), gradients):
            m = w_opt.state[w].get("momentum_buffer", 0.)
            w = w - lr * (momentum * m + g + weight_decay * w)

    # calculate unrolled loss on validation data
    # keep gradients for model here for compute hessian
    mutator.reset()
    output = model(val_x)
    
    a_loss = criterion(output, val_y)
    w_model, w_ctrl = tuple(model.parameters()), tuple(mutator.parameters())
    w_grads = torch.autograd.grad(a_loss, w_model + w_ctrl)
    d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

    # compute hessian and final gradients
    _restore_weights(model, backup_params)
    norm = torch.cat([w.view(-1) for w in d_model]).norm()
    eps = 0.01 / norm
    if norm < 1E-8:
        print("In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.", norm.item())

    dalphas = []
    for e in [eps, -2. * eps]:
        # w+ = w + eps*dw`, w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(model.parameters(), d_model):
                p += e * d

        mutator.reset()
        output = model(train_x)
        a_loss = criterion(output, train_y)
        dalphas.append(torch.autograd.grad(a_loss, mutator.parameters()))

    dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
    hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]

    # hessian = _compute_hessian(backup_params, d_model, train_x, train_y)
    with torch.no_grad():
        for param, d, h in zip(w_ctrl, d_ctrl, hessian):
            # gradient = dalpha - lr * hessian
            param.grad = d - lr * h

    # restore weights
    _restore_weights(model, backup_params)

def _restore_weights(model, backup_params):
    with torch.no_grad():
        for param, backup in zip(model.parameters(), backup_params):
            param.copy_(backup)


def train(train_loader, val_loader, model, mutator, criterion, w_opt, a_opt, is_rolled, device, epoch):
    model.train()
    for batch_idx, ((train_x, train_y), (val_x, val_y)) in enumerate(zip(train_loader, val_loader)):
        train_x, train_y = train_x.type(torch.float).to(device, non_blocking=True), train_y.type(torch.float).to(device, non_blocking=True)
        val_x, val_y = val_x.to(device, non_blocking=True), val_y.type(torch.float).to(device, non_blocking=True)

        # update architecture
        if a_opt is not None:
            a_opt.zero_grad(set_to_none=True)
            if is_rolled:
                unrolled_backward()
            else:
                mutator.reset()
                output = model(val_x)
                # print(output.shape, val_y.shape)
                a_loss = criterion(output, val_y)
                a_loss.backward()
            a_opt.step()

        # update weights
        w_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            mutator.reset()
        output = model(train_x)
        w_loss = criterion(output, train_y)
        w_loss.backward()
        w_opt.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(train_x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), w_loss.item()))

def validate(loader, model, criterion, device, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.type(torch.float).to(device, non_blocking=True), y.type(torch.float).to(device, non_blocking=True)
            output = model(x)
            loss = criterion(output, y)
            test_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    test_acc = 100. * correct / len(loader.dataset)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), test_acc))
    return test_loss, test_acc


# Main training loop
criterion = torch.nn.CrossEntropyLoss()
history = {}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
model = net.to(device)
mutator = dm.to(device)
is_rolled = False
for epoch in tqdm(range(1, 50)):
    train(train_loader, val_loader, model, mutator, criterion, network_opt, architecture_opt, is_rolled, device, epoch)
    mask = mutator.export()
    mutator.sample_by_mask(mask)
    # arch = model.arch
    val_loss, val_acc = validate(test_loader, model, criterion, device, verbose=True)
    print(f"acc={val_acc} loss={val_loss}")
    # print(f"{arch} acc={val_acc} loss={val_loss}")
    torch.save(model.state_dict(), 'da_mobilenet3d_supernet.pt')
    torch.save(mutator.state_dict(), 'da_mobilenet3d_darts.pt')