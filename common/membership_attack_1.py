from sklearn.svm import SVC
import torch
import tqdm
import numpy as np

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False)
    prob = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, leave=False)):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).cpu().numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, model)
    clf = SVC(C=3, gamma='auto', kernel='rbf')
    # clf = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='multinomial')
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()


def plot_entropy_dist(model, ax, title):
    train_loader_full, test_loader_full = datasets.get_loaders(dataset, batch_size=100, seed=0, augment=False,
                                                               shuffle=False)
    indexes = np.flatnonzero(np.array(train_loader_full.dataset.targets) == class_to_forget)
    replaced = np.random.RandomState(0).choice(indexes, size=100 if num_to_forget == 100 else len(indexes),
                                               replace=False)
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(train_loader_full, test_loader_full, model, replaced)
    sns.distplot(np.log(X_r[Y_r == 1]).reshape(-1), kde=False, norm_hist=True, rug=False, label='retain', ax=ax)
    sns.distplot(np.log(X_r[Y_r == 0]).reshape(-1), kde=False, norm_hist=True, rug=False, label='test', ax=ax)
    sns.distplot(np.log(X_f).reshape(-1), kde=False, norm_hist=True, rug=False, label='forget', ax=ax)
    ax.legend(prop={'size': 14})
    ax.tick_params(labelsize=12)
    ax.set_title(title, size=18)
    ax.set_xlabel('Log of Entropy', size=14)
    ax.set_ylim(0, 0.4)
    ax.set_xlim(-35, 2)


def membership_attack(retain_loader, forget_loader, test_loader, model):
    prob = get_membership_attack_prob(retain_loader, forget_loader, test_loader, model)
    print("Attack prob: ", prob)
    return prob


# %%
attack_dict = {}
# %%
attack_dict['Original'] = membership_attack(retain_loader, forget_loader, test_loader_full, model)
# %%
attack_dict['Retrain'] = membership_attack(retain_loader, forget_loader, test_loader_full, model0)
# %%
attack_dict['NTK'] = membership_attack(retain_loader, forget_loader, test_loader_full, model_scrub)
# %%
attack_dict['Fisher'] = membership_attack(retain_loader, forget_loader, test_loader_full, modelf)
# %%
attack_dict['Finetune'] = membership_attack(retain_loader, forget_loader, test_loader_full, model_ft)
# %%
attack_dict['Fisher_NTK'] = membership_attack(retain_loader, forget_loader, test_loader_full, model_scrubf)