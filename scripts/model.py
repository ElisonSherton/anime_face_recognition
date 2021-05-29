class network(nn.Module):
    def __init__(self, output_dim = 256):
        '''
        Define network parameters and architecture
        '''
        super().__init__()
        self.body = create_body(resnet50, pretrained = True)
        fe = num_features_model(self.body)
        self.feat_extractor = nn.Linear(in_features = fe * 2, out_features = output_dim)
    
    def forward(self, x):
        '''
        Define the forward pass behaviour of the model here
        '''
        conv_op = self.body(x)
        pooled = Flatten()(AdaptiveConcatPool2d()(conv_op))
        features = self.feat_extractor(pooled)
        return features