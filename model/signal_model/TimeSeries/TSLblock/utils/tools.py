import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, reportfun=print):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        #reportfun('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0,report_fun = print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.info = report_fun

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path,epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path,epoch):
        if self.verbose:
            self.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({"model":model.state_dict(),"epoch":epoch}, path)
        self.val_loss_min = val_loss

 
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


colormap = ['r', 'b', 'p', 'y']


def multistep_demo(multistep_result, multistep_ground, seq_len, steps, step_length, sample_id=0, feature_id=0, name='./demo.png'):
    plot_data = []
    pred = multistep_result[sample_id, :, feature_id]
    real = multistep_ground[sample_id, :, feature_id]
    inp = multistep_result[sample_id, :seq_len, feature_id]
    x = np.arange(len(pred))
    x_real = x[seq_len:]
    y_real = real[seq_len:]
    x_inp = x[:seq_len]
    y_inp = inp
    fig = plt.figure()
    line1 = []
    # result_length    = seq_len + (steps-1)*step_length + pred_len 
    # the goal for below code is try to color different intervel
    #for i, x_pred in enumerate(np.stack([x[seq_len:], pred[seq_len:]], 1).reshape(steps, step_length, 2)):
    start_to_end_list = list(seq_len + np.arange(steps)*step_length) + [len(pred)]
    for i,(start,end)  in enumerate(zip(start_to_end_list[:-1],start_to_end_list[1:])):
        t =  x[start:end]
        p = pred[start:end]
        color = colormap[i % len(colormap)]
        plt.plot(t, p, color)
        for pos, val in zip(t, p):
            line1.append([pos, val, color, 'pred'])
    line2 = []
    plt.plot(x_real, y_real, 'black', label='groundt')
    for pos, val in zip(x_real, y_real):
        line2.append([pos, val, 'black', 'groundt'])
    line3 = []
    plt.plot(x_inp, y_inp, 'g', label='input')
    for pos, val in zip(x_inp, y_inp):
        line3.append([pos, val, 'g', 'input'])

    plt.legend()
    plt.title(f'test demo for sample:{sample_id} feature:{feature_id}')
    plt.xlabel('time_to_predict')
    plt.savefig(name, bbox_inches='tight')
    return line1, line2, line3
# multistep_demo(multistep_result, multistep_ground,seq_len,steps, step_length)


def multistep_error_plot(multistep_result, multistep_ground, seq_len, steps, step_length, name='./test.png'):
    """
    Results visualization
    """
    delta = ((multistep_result - multistep_ground)**2)
    res  = delta[:, seq_len+(steps-1)*step_length:] #(B, pred_len, D)
    res  = res.mean() # (1,)
    delta = delta[:, seq_len:seq_len+(steps-1)*step_length]
    delta = delta.reshape(delta.shape[0], steps-1, step_length, -1)
    delta = list(delta.mean(axis=(0, 2, 3)))# (steps,)
    delta = np.array(delta + [res])
    if len(delta) == 0:return [[0, 0]]
    delta_line = np.broadcast_to(delta[:, None], (steps, step_length)).flatten()
    delta_line = np.concatenate([np.zeros(seq_len), delta_line])

    fig = plt.figure()
    plt.plot(delta_line)
    line = [[x, y] for x, y in zip(np.arange(len(delta_line)), delta_line.tolist())]
    plt.title('mean square error')
    plt.xlabel('time_to_predict')
    plt.savefig(name, bbox_inches='tight')
    return line
