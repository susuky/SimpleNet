
import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


DEBUG =  __name__ == '__main__'


def init_weight(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight)


class TimmFeatureExtractor(nn.Module):
    '''
    Note: Only cnn-based models are available
    
    Args:
        model_name: str
        layers: list[int], 
    '''
    def __init__(self, model_name='resnet18', layers=[2, 3], **kwargs):
        super(TimmFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.feature_extractor = timm.create_model(model_name, 
                                                   features_only=True, 
                                                   out_indices=layers, 
                                                   **kwargs)
    def forward(self, x):
        with torch.no_grad():
            return self.feature_extractor(x)
    
    @property
    def info(self):
        return self.feature_extractor.feature_info.get_dicts()

    @property
    def channels(self):
        return self.feature_extractor.feature_info.channels()
    
    @property
    def module_names(self):
        return self.feature_extractor.feature_info.module_name()
    
    
class Backbone(TimmFeatureExtractor):
    def __init__(self, model_name='resnet18', layers=[2, 3], **kwargs):
        super(Backbone, self).__init__(model_name, layers, **kwargs)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False        
    
    @property
    def out_dims(self):
        return sum(self.channels)
    

if DEBUG:
    backbone = Backbone('resnet18', layers=[2, 3]) #'wide_resnet50_2'
    x = torch.randn(1, 3, 224, 224)
    feats = backbone(x)
    print('Module names: ',  backbone.module_names)
    print('Feature infos: ', backbone.info)
    print('Embedding dims: ', backbone.out_dims)
    print('features\' shape: ')
    for i in feats:
        print(i.shape)


class Adaptor(torch.nn.Module):
    def __init__(self, h=768, out_dims=None):
        super().__init__()
        self.h = h
        self.out_dims = out_dims if out_dims is not None else h
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.bs, self.emb_size, self.height, self.width = None, None, None, None        
        self.linear = nn.Sequential(
            torch.nn.Linear(self.h, self.out_dims),
            #torch.nn.BatchNorm1d(self.out_dims),
            #torch.nn.Tanh(),
        )
        self.apply(init_weight)

    def patchify(self, x):
        k, p, s = 3, 1, 1
        unfold = F.unfold(x, kernel_size=k, padding=p, stride=s)
        unfold = unfold.reshape(x.shape[0], x.shape[1], k, k, -1)
        return unfold.permute(0, 4, 3, 1, 2)

                
    def generate_embedding(self, features) -> torch.Tensor:
        '''Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        '''
        embeddings = features[0]
        for feature in features[1:]:
            layer_embedding = F.interpolate(feature, size=embeddings.shape[-2:], mode='bilinear')
            embeddings = torch.cat((embeddings, layer_embedding), 1)
        return embeddings
    
    def reshape_embedding(self, embedding):
        '''
        (BS, Embedding, H, W) -> (BS, H, W, Embedding)
        '''
        self.bs, self.emb_size, self.h0, self.w0 = embedding.shape
        #return embedding.permute(0, 2, 3, 1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, self.emb_size)    
    
    def forward(self, feats):
        feats = [self.feature_pooler(feat) for feat in feats]
        embedding = self.generate_embedding(feats)
        embedding = self.reshape_embedding(embedding)
        x = self.linear(embedding)
        return x


if DEBUG:
    adaptor = Adaptor(backbone.out_dims)
    embedding = adaptor(feats)
    print('embedding shape: ', embedding.shape)


class Generator(nn.Module):
    def __init__(self, std=0.015):
        super(Generator, self).__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            # Note: If you use `x += torch.normal(0, self.std, x.shape)`,
            # it will affect the original embedding vector `x`.
            x = x + torch.normal(0, self.std, x.shape).to(x.device)
        return x
    
    
class Discriminator(torch.nn.Module):
    def __init__(self, h=1536, out_dims=None):
        super().__init__()
        self.out_dims = out_dims if out_dims is not None else h
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(h, self.out_dims),
            torch.nn.BatchNorm1d(self.out_dims),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Linear(self.out_dims, 1, False),
        )
        self.apply(init_weight)

    def forward(self, x):
        if self.training:
            return self.fc(x)
        return -self.fc(x)
    

if DEBUG:
    generator = Generator()
    discirminator = Discriminator(adaptor.out_dims)

    new_embedding = generator(embedding)
    score = discirminator(new_embedding)
    print('score shape: ', score.shape)


class ComputeLoss:
    def __init__(self, thr=0.5):
        self.thr = thr
        
    def __call__(self, embedding, generator, discriminator):
        nums = len(embedding)
        fake_embedding = generator(embedding)
        scores = discriminator(torch.cat([embedding, fake_embedding], dim=0))
        true_scores, fake_scores = torch.split(scores, nums, dim=0)
        
        assert (fake_embedding != embedding).any()
        true_loss = torch.clip(self.thr - true_scores, min=0)
        fake_loss = torch.clip(self.thr + fake_scores, min=0)
        #loss = (true_loss + fake_loss).mean()
        loss = true_loss.mean() + fake_loss.mean()
        
        p_true = (true_scores.detach() >= self.thr).float().mean()
        p_fake = (fake_scores.detach() < -self.thr).float().mean()
        return loss, p_true, p_fake
    

class AnomalyMapGenerator(nn.Module):
    def __init__(self, sigma=4.0):
        super(AnomalyMapGenerator, self).__init__()
        self.ks = 2 * int(4.0 * sigma + 0.5) + 1
        self.sigma = 4.0
        self.blur = T.GaussianBlur(self.ks, self.sigma)
        
    def forward(self, patch_scores, img_size=None):
        if img_size is not None:
            patch_scores = F.interpolate(patch_scores, size=tuple(img_size[:2]))
        anomaly_map = self.blur(patch_scores)
        anomaly_map = anomaly_map.mean(axis=1).squeeze(axis=1)
        return anomaly_map
    

class SimpleNet(nn.Module):
    '''
    model_name: str. 
        Use TimmFeatureExtractor to get the feature map, 
        and only cnn-based models are available.
    '''
    def __init__(self, model_name='resnet18', layers=[2, 3], std=0.015, **kwargs):
        super(SimpleNet, self).__init__()
        self.backbone = Backbone(model_name, layers, **kwargs)
        self.adaptor = Adaptor(self.backbone.out_dims)
        self.generator = Generator(std)
        self.discriminator = Discriminator(self.adaptor.out_dims, 1024)
        self.anomaly_map_generator = AnomalyMapGenerator()
        # for anomaly_map localization
        self.reset_minmax()
        self.threshold = 0.1

    def reset_minmax(self):
        self.max = float('-inf')
        self.min = float('inf')

    @property
    def bs(self):
        return self.adaptor.bs
    
    def forward(self, x, track=False):
        with torch.no_grad():
            bs, _, height, width = x.shape
            x = self.backbone(x)
            embeddings = self.adaptor(x)                  # (bs*h*w, emb_size)
            patch_scores = self.discriminator(embeddings) # (bs*h*w, 1)
            image_scores = patch_scores.view(self.bs, -1).amax(axis=1)
            if track:
                self.max = max(self.max, image_scores.max().item())
                self.min = min(self.min, image_scores.min().item())

            anomaly_map = patch_scores.view(bs, 
                                            1, 
                                            self.adaptor.h0, 
                                            self.adaptor.w0)
            anomaly_map = self.anomaly_map_generator(anomaly_map, 
                                                     img_size=(height, width))
        return anomaly_map, image_scores
    
if DEBUG:
    #simplenet = SimpleNet('efficientnet_b0', pretrained=True)
    simplenet = SimpleNet('resnet18', pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    anomaly_map, image_scores = simplenet(x)
    assert anomaly_map.shape == (1, 224, 224)
    assert image_scores.shape == (1,)


def save_model(model, path='models/model.pth'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    state_dict = {}
    state_dict['model'] = model.state_dict()

    metadata = {}
    metadata['max'] = model.max
    metadata['min'] = model.min
    state_dict['metadata'] = metadata
    torch.save(state_dict, path)


def load_model(model, path='models/model.pth'):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    model.max = state_dict['metadata']['max']
    model.min = state_dict['metadata']['min']
    return model

