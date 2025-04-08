


def get_sum_score(path_list, score_list):

    new_path_list = []
    new_score_list = []
    for i in range(len(path_list)):
        img = path_list[i]
        if img not in new_path_list:
            new_path_list.append(img)
            img_indices = [index for index, element in enumerate(path_list) if img == element]
            score = 0
            for ii in img_indices:
                score += score_list[ii]
            new_score_list.append(score)

    return new_path_list, new_score_list
