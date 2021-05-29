from fastai.vision.all import *

def get_character_items(item):
    images_data_path = '/home/vinayak/AnimeFaceDataset'
    x = f'{images_data_path}' + '/' + item.images
    y = item.images.apply(lambda x: x.split('/')[0])
    return (x, y)

def create_databunch(csv_path_name, bs = 16):
    
    df = pd.read_csv(csv_path_name)
    
    # Get the indices of validation set images
    validation_indices = df[df.is_valid == 'valid'].index.values
    
    characters = DataBlock.from_columns(
                                   # Specify the input and output types
                                   blocks = (ImageBlock, CategoryBlock),
    
                                   # Specify how to read from the datframe which image is train & which is valid
                                   get_items = get_character_items,
    
                                   # Specify the train and validation split indices
                                   splitter = IndexSplitter(validation_indices),
    
                                   # Specify item transformations
                                   item_tfms = RandomResizedCrop(300),
    
                                   # Specify batch transformations
                                   batch_tfms = [*aug_transforms(size = 225), 
                                                 Normalize.from_stats(*imagenet_stats)]
                                   )
    
    # Create dataloader from the databunch defined above
    dls = characters.dataloaders(df, bs = bs)
    
    return dls