import argparse
import torch
import numpy as np
import utils1
from mutual_information import mutual_information
from dataset import Dateset_mat
from tqdm import trange
from model import Model, UD_constraint
import torch.distributions.normal as normal
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default=r'nus-wide', type=str)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--num_epochs", type=int, default=500)
config = parser.parse_args()

Dataset = Dateset_mat(config.dataset_root)
dataset = Dataset.getdata()
label = np.array(dataset[2]) - 1
label = np.squeeze(label)
cluster_num = max(label) + 1
print("clustering number: ", cluster_num)
img = torch.tensor(dataset[0], dtype=torch.float32).to(device)
txt = torch.tensor(dataset[1], dtype=torch.float32).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)

prior_loc = torch.zeros(img.size(0), 100)
prior_scale = torch.ones(img.size(0), 100)
prior = normal.Normal(prior_loc, prior_scale)


def run():
    max_ACC = 0
    model = Model(cluster_num * 10, cluster_num)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in trange(config.num_epochs):
        model.train()
        model.zero_grad()
        x_img_overC, x_img_C, encoder_img = model(img)
        x_txt_overC, x_txt_C, encoder_txt = model(txt)
        loss1 = mutual_information(x_img_C, x_txt_C).to(device) \
                + mutual_information(x_img_overC, x_txt_overC).to(device)  # L1:preserve the shared information

        z_img, z_txt = model.encoder(img), model.encoder(txt)
        z1, z2, prior_sample = z_img.rsample().cpu(), z_txt.rsample().cpu(), prior.sample()
        z1, z2 = F.log_softmax(z1, dim=-1), F.log_softmax(z2, dim=-1)
        prior_sample = F.softmax(prior_sample, dim=-1)
        skl1 = torch.nn.functional.kl_div(z1, prior_sample).to(device)
        skl2 = torch.nn.functional.kl_div(z2, prior_sample).to(device)
        loss2 = skl1 + skl2                              # L2:eliminate the superfluous information

        if epoch % 5 == 0:
            with torch.no_grad():
                UDC = UD_constraint(model, img)
                UDC = UDC.to(device)
        loss3 = criterion(x_img_C, UDC)/2                # uniform distribution constraint

        loss_g = loss1 + loss2 + loss3
        loss_g.backward(retain_graph=True)
        optimiser.step()

        if epoch % 10 == 0:
            model.eval()
            _, x_out, _ = model(img)

            pre_label = np.array(x_out.cpu().detach().numpy())
            pre_label = np.argmax(pre_label, axis=1)

            acc = utils1.metrics.acc(pre_label, label)
            nmi = utils1.metrics.nmi(pre_label, label)

            if acc > max_ACC:
                max_ACC = acc
            print(" epoch %d loss1 %.3f acc %.3f nmi %.3f" % (epoch, loss_g, acc, nmi))


if __name__ == '__main__':
    run()

