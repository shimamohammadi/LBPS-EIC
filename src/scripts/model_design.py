import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Architecture(nn.Module):

    def __init__(self, batch_size, rate=0.2):
        super(Architecture, self).__init__()

        self.rate = rate

        self.flatten = nn.Flatten(2, 3)
        self.flatten2 = nn.Flatten()

        self.fc1_score = nn.Linear(36864, 512)
        self.fc2_score = nn.Linear(512, 1)

        self.fc1_weight = nn.Linear(36864, 512)
        self.fc2_weight = nn.Linear(512, 1)

        self.conv_layer_1 = nn.Conv2d(
            in_channels=64, out_channels=96, kernel_size=1)

        self.conv_layer_2 = nn.Conv2d(
            in_channels=128, out_channels=96, kernel_size=1)

        self.conv_layer_3 = nn.Conv2d(
            in_channels=256, out_channels=96, kernel_size=1)

        self.conv_layer_4 = nn.Conv2d(
            in_channels=512, out_channels=96, kernel_size=1)

        self.resnet = torchvision.models.resnet.resnet34(
            weights="ResNet34_Weights.DEFAULT")

        self.ref_score_subtract = nn.Linear(1, 1)

        self.batch_size = batch_size
        self.dropout = nn.Dropout(self.rate)

    def feature_extraction(self, x):
        """
            Feature Extraction module
        """

        conv1 = self.resnet.conv1(x)
        bn1 = self.resnet.bn1(conv1)
        bn1_relu = self.resnet.relu(bn1)
        maxpool = self.resnet.maxpool(bn1_relu)

        layer_1 = self.resnet.layer1(maxpool)
        layer_2 = self.resnet.layer2(layer_1)
        layer_3 = self.resnet.layer3(layer_2)
        layer_4 = self.resnet.layer4(layer_3)

        return layer_1, layer_2, layer_3, layer_4

    def channel_reduction(self, layer_1, layer_2, layer_3, layer_4):

        layer_1_reduced = self.conv_layer_1(layer_1)
        layer_2_reduced = self.conv_layer_2(layer_2)
        layer_3_reduced = self.conv_layer_3(layer_3)
        layer_4_reduced = self.conv_layer_4(layer_4)

        return layer_1_reduced, layer_2_reduced, layer_3_reduced, layer_4_reduced

    def measure_distnace(self, layer_1_dis, layer_2_dis, layer_3_dis, layer_4_dis, layer_1_ref, layer_2_ref, layer_3_ref, layer_4_ref):
        diff_layer_1 = layer_1_ref - layer_1_dis
        diff_layer_2 = layer_2_ref - layer_2_dis
        diff_layer_3 = layer_3_ref - layer_3_dis
        diff_layer_4 = layer_4_ref - layer_4_dis

        return diff_layer_1, diff_layer_2, diff_layer_3, diff_layer_4

    def bilinear_calculation(self, reduced_layer_1, reduced_layer_2, reduced_layer_3, reduced_layer_4):
        reduced_layer_1_bi = self._bilinearpool(reduced_layer_1)
        reduced_layer_2_bi = self._bilinearpool(reduced_layer_2)
        reduced_layer_3_bi = self._bilinearpool(reduced_layer_3)
        reduced_layer_4_bi = self._bilinearpool(reduced_layer_4)

        features_distance = torch.cat((self.flatten2(reduced_layer_1_bi), self.flatten2(
            reduced_layer_2_bi), self.flatten2(reduced_layer_3_bi), self.flatten2(reduced_layer_4_bi)), 1)

        return features_distance

    def _bilinearpool(self, x):
        """ This Function taken from UNIQUE
        """
        batchSize, C, h, w = x.data.shape
        x = x.reshape(batchSize, C, h * w)
        x = 1. / (h * w) * x.bmm(x.transpose(1, 2))
        return x

    def Quality_assessment(self, image_A_patches, layer_1_ref, layer_2_ref, layer_3_ref, layer_4_ref):

        layer_1_dis, layer_2_dis, layer_3_dis, layer_4_dis = self.feature_extraction(
            image_A_patches)

        diff_layer_1, diff_layer_2, diff_layer_3, diff_layer_4 = self.measure_distnace(
            layer_1_dis, layer_2_dis, layer_3_dis, layer_4_dis, layer_1_ref, layer_2_ref, layer_3_ref, layer_4_ref)

        reduced_layer_1, reduced_layer_2, reduced_layer_3, reduced_layer_4 = self.channel_reduction(
            diff_layer_1, diff_layer_2, diff_layer_3, diff_layer_4)

        features_distance = self.bilinear_calculation(
            reduced_layer_1, reduced_layer_2, reduced_layer_3, reduced_layer_4)

        mean_fc_1_output = F.relu(self.fc1_score(features_distance))
        mean_fc_1_output = self.dropout(mean_fc_1_output)
        mean = self.fc2_score(mean_fc_1_output)

        sigma_fc_1_output = F.relu(self.fc1_weight(features_distance))
        sigma_fc_1_output = self.dropout(sigma_fc_1_output)
        sigma = self.fc2_weight(sigma_fc_1_output)
        sigma = nn.functional.softplus(sigma)

        return mean, sigma

    def forward(self, ref, dis_a, dis_b):
        layer_1_ref, layer_2_ref, layer_3_ref, layer_4_ref = self.feature_extraction(
            ref)

        mean_a, sigma_a = self.Quality_assessment(
            dis_a, layer_1_ref, layer_2_ref, layer_3_ref, layer_4_ref)

        mean_b, sigma_b = self.Quality_assessment(
            dis_b, layer_1_ref, layer_2_ref, layer_3_ref, layer_4_ref)

        mean_diff = mean_a - mean_b

        # y_var is data uncertainty
        y_var = sigma_a * sigma_a + sigma_b * sigma_b + 1e-8
        p = 0.5 * (1 + torch.erf(mean_diff / torch.sqrt(2 * y_var.detach())))

        return p
