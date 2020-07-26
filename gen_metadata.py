from fastai2.basics           import *
from fastai2.medical.imaging  import *

main_dir=Path("/home/tkrsh/osic-main/")


files=[]

for dirname, _, filenames in os.walk(main_dir):
    _.sort()
    for filename in (filenames):
        files.append(os.path.join((dirname), filename))

files=[x for x in files if '.csv' not in x]

train_images= [Path(x) for x in files if 'train'  in x]
test_images = [Path(x) for x in files if 'test'   in x]


dcm_metadata_train=pd.DataFrame.from_dicoms(train_images,px_summ=True)
dcm_metadata_test=pd.DataFrame.from_dicoms(test_images,px_summ=True)

dcm_metadata_train.to_csv("dcm_metadata_train.csv")
dcm_metadata_test.to_csv("dcm_metadata_test.csv")