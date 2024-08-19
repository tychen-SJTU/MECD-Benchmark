import torch
import torch.nn.functional as F



def InfoCE_Loss(unmasked_feature, masked_feature_list, relation, temp=1.0):
    similarity_matrix_list = []
    for masked_feature in masked_feature_list:
        similarity_matrix_list.append(F.cosine_similarity(unmasked_feature, masked_feature, dim=1).mean())
    positive_sum, total_sum = 0, 0
    if len(relation) != len(masked_feature_list):
        print("error in marking")
    for i in range(0, len(masked_feature_list)):
        if relation[i] == 0:
            positive_sum += (torch.exp(similarity_matrix_list[i]/temp))
        total_sum += (torch.exp(similarity_matrix_list[i]/temp))
    positive_sum = torch.tensor(positive_sum)
    total_sum = torch.tensor(total_sum)
    loss = -torch.log(positive_sum.clone().detach().cuda() / total_sum.clone().detach().cuda())
    return loss.mean(), similarity_matrix_list
