def check_stopping_criteria(optimizer, dataset, alpha, best_accuracy, rnd):
    if(dataset == 'CIFAR10'):
        if(rnd >= 1000 and best_accuracy <= 20):
            return True
        # fedadam
        elif(optimizer == 'fedadam' and alpha == 0.05 and rnd >= 30 and best_accuracy <= 40):
            return True
        # fednova | alpha >= 0.1
        elif(optimizer == 'fednova' and alpha >= 0.1 and rnd >= 30 and best_accuracy <= 60):
            return True
        # fednova | alpha < 0.1
        elif(optimizer == 'fednova' and alpha < 0.1 and rnd >= 30 and best_accuracy <= 22):
            return True
    return False