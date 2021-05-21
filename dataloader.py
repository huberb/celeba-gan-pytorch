from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class FilteredDataset(Dataset):

    def __init__(self, img_size,
                 attribute="Smiling", amount=None, root="./data"):
        super(FilteredDataset, self).__init__()

        CelebA(download=True, root=root)
        self.label_file_path = f"{root}/celeba/list_attr_celeba.txt"
        self.img_folder_path = f"{root}/celeba/img_align_celeba"

        self.transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            ])

        label_lines = open(self.label_file_path).readlines()
        label_list = label_lines[1].split(" ")
        target_idx = label_list.index(attribute) + 1

        self.files = []
        for index, line in enumerate(label_lines[2:]):
            line = line.replace("  ", " ")
            line = line.replace(" ", " ")
            line = line.replace("\n", "")
            line = line.split(" ")
            if int(line[target_idx]) == 1:
                self.files.append(line[0])
            if amount is not None and len(self.files) == amount:
                break

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        img = Image.open(f"{self.img_folder_path}/{file_name}")
        return self.transform(img)
