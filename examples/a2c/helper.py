from datetime import datetime
import pytz
import torch

total_time = 0
last_save = 0

def gen_data(x):
    return [f(x.float()).item() for f in [torch.mean, torch.median, torch.min, torch.max, torch.std]]

def format_time(f):
    return datetime.fromtimestamp(f, tz=pytz.utc).strftime('%H:%M:%S.%f s')

def callback(args, model, frames, iter_time, rewards, lengths,
             value_loss, policy_loss, entropy, csv_writer, csv_file):
    global last_save, total_time

    if not hasattr(args, 'num_steps_per_update'):
        args.num_steps_per_update = args.num_steps

    total_time += iter_time
    fps = (args.world_size * args.num_steps_per_update * args.num_ales) / iter_time
    lmean, lmedian, lmin, lmax, lstd = gen_data(lengths)
    rmean, rmedian, rmin, rmax, rstd = gen_data(rewards)

    if frames >= last_save:
        last_save += args.save_interval

        # torch.save(model.state_dict(), args.model_name)

        if csv_writer and csv_file:
            csv_writer.writerow([frames, fps, total_time,
                                 rmean, rmedian, rmin, rmax, rstd,
                                 lmean, lmedian, lmin, lmax, lstd,
                                 entropy, value_loss, policy_loss])
            csv_file.flush()

    str_template = '{fps:8.2f}f/s, ' \
                   'min/max/mean/median reward: {rmin:5.1f}/{rmax:5.1f}/{rmean:5.1f}/{rmedian:5.1f}, ' \
                   'entropy/value/policy: {entropy:6.4f}/{value:6.4f}/{policy: 6.4f}'

    return str_template.format(fps=fps, rmin=rmin, rmax=rmax, rmean=rmean, rmedian=rmedian,
                               entropy=entropy, value=value_loss, policy=policy_loss)
