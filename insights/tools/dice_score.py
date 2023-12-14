import torch

class DiceCalculator:
    def __init__(self, gt, pred, device, n_classes=3):
        self.gt = gt
        self.pred = pred
        self.device = device
        self.n_classes = n_classes
        self.calculate_dice_score()
        self.fix_dice_score()

    def one_hot(self, labelmap, eps=1e-6):
        shp = labelmap.shape
        #print(shp)
        one_hot = torch.zeros((shp[0], self.n_classes) + shp[1:], device=self.device, dtype=torch.int64)
        #print(one_hot.shape)
        return one_hot.scatter_(1, labelmap.unsqueeze(1), 1.0) + eps

    def calculate_dice_score(self, eps: float = 1e-6):
        """
        Adapted from https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/dice.html#dice_loss
        """
        input_one_hot = self.one_hot(self.pred)
        #print("Number of unique classes in ground truth:", len(torch.unique(self.gt)))
        #print(self.gt.shape)
        #print(self.pred.shape)
        #print("Number of unique classes in prediction:", len(torch.unique(self.pred)))

        target_one_hot: torch.Tensor = self.one_hot(self.gt)
        dims = (2, 3)
        intersection = torch.sum(input_one_hot * target_one_hot, dims)
        cardinality = torch.sum(input_one_hot + target_one_hot, dims)
        dice_score = 2.0 * intersection / (cardinality + eps)
        self.dice_score = dice_score

    def fix_dice_score(self):
        for idx, (gt_img, pred_img) in enumerate(zip(self.gt, self.pred)):
            labels_gt = gt_img.unique()
            labels_pred = pred_img.unique()
            # if there are not 3 labels in the ground truth, we need to change the calculation of the dice_score mean
            if len(labels_gt) == len(labels_pred) and len(labels_gt) == 2:
                if all(torch.eq(labels_gt.to(torch.int8), labels_pred.to(
                        torch.int8))):  # make sure that we only consider matching classes, so the calculation will not change if GT is  only weed and the prediction is only sorghum
                    # get the class that is missing in the ground truth
                    all_labels = torch.tensor([0, 1, 2], device=self.device)
                    # Create a tensor to compare all values at once
                    compareview = labels_gt.repeat(all_labels.shape[0], 1).T
                    # Non Intersection
                    label_idx = all_labels[(compareview != all_labels).T.prod(1) == 1]
                    # replace the values of the dice_score with 'nan' at idx and label_idx cell
                    self.dice_score[idx, label_idx] = torch.nan

            elif len(labels_gt) == len(labels_pred) and len(labels_gt) == 1:  # if there is only background in GT
                if all(torch.eq(labels_gt.to(torch.int8), labels_pred.to(torch.int8))):
                    if labels_gt == 0:
                        self.dice_score[idx, 1] = torch.nan
                        self.dice_score[idx, 2] = torch.nan
        return