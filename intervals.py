from numpy import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import os


def find_best_interval(xs, ys, k):
    assert all(array(xs) == array(sorted(xs))), "xs must be sorted!"

    xs = array(xs)
    ys = array(ys)
    m = len(xs)
    P = [[None for j in range(k+1)] for i in range(m+1)]
    E = zeros((m+1, k+1), dtype=int)

    # Calculate the cumulative sum of ys, to be used later
    cy = concatenate([[0], cumsum(ys)])

    # Initialize boundaries:
    # The error of no intervals, for the first i points
    E[:m+1, 0] = cy

    # The minimal error of j intervals on 0 points - always 0. No update needed.        

    # Fill middle
    for i in range(1, m+1):
        for j in range(1, k+1):
            # The minimal error of j intervals on the first i points:

            # Exhaust all the options for the last interval. Each interval boundary is marked as either
            # 0 (Before first point), 1 (after first point, before second), ..., m (after last point)
            options = []
            for l in range(0, i+1):
                next_errors = E[l, j-1] + (cy[i]-cy[l]) + concatenate([[0], cumsum((-1)**(ys[arange(l, i)] == 1))])
                min_error = argmin(next_errors)
                options.append((next_errors[min_error], (l, arange(l, i+1)[min_error])))

            E[i, j], P[i][j] = min(options)

    # Extract best interval set and its error count
    best = []
    cur = P[m][k]
    for i in range(k, 0, -1):
        best.append(cur)
        cur = P[cur[0]][i-1]       
        if cur is None:
            break 
    best = sorted(best)
    besterror = E[m, k]

    # Convert interval boundaries to numbers in [0,1]
    exs = concatenate([[0], xs, [1]])
    representatives = (exs[1:]+exs[:-1]) / 2.0
    intervals = [(representatives[l], representatives[u]) for l, u in best]

    return intervals, besterror


def create_training_set(m, distribution):
    training_set = []
    for i in range(m):
        x = random.uniform(0, 1)
        for interval in distribution:
            if interval[0][0] <= x <= interval[0][1]:
                y = random.binomial(1, interval[1])
                break
        else:
            assert False, "x not in [0,1]"
        training_set.append((x, y))
    training_set.sort(key=lambda x: x[0])
    xs = [x[0] for x in training_set]
    ys = [y[1] for y in training_set]
    return xs, ys


def empirical_error(xs, ys, intervals):
    errors = 0
    for i in range(len(xs)):
        if len([True for interval in intervals if interval[0] <= xs[i] <= interval[1]]) > 0:
            if ys[i] == 0:
                errors += 1
        elif ys[i] == 1:
            errors += 1
    return float(errors) / len(xs)


def true_error(intervals, distribution):
    intervals.sort(key=lambda x: x[0])
    current_point = 0
    space_partition = []
    for interval in intervals:
        if interval[0] == current_point:
            space_partition.append(((interval[0], interval[1]), 1))
            current_point = interval[1]
        else:
            space_partition.append(((current_point, interval[0]), 0))
            space_partition.append(((interval[0], interval[1]), 1))
            current_point = interval[1]
    if space_partition[-1][0][1] < 1:
        space_partition.append(((space_partition[-1][0][1], 1), 0))
    loss = 0
    for interval in space_partition:
        loss += interval_true_error(interval[0], interval[1], distribution)

    return loss


def interval_true_error(interval, type, distribution):
    loss = 0
    start, end = interval
    for part in distribution:
        interval_loss = part[1] if type == 0 else 1 - part[1]
        if end <= part[0][1]:
            loss += (end - start) * interval_loss
            return loss
        elif start < part[0][1]:
            loss += (part[0][1] - start) * interval_loss
            start = part[0][1]
    return loss


def a_part(dir_path, distribution):
    lines = [0.25, 0.5, 0.75]
    k = 2
    xs, ys = create_training_set(100, distribution)
    intervals, besterror = find_best_interval(xs, ys, k)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys, s=2)
    for line in lines:
        ax.plot((line, line), (0, 1), linewidth=1, marker='', color='r')
    first_interval = True
    for interval in intervals:
        if first_interval:
            ax.plot((interval[0], interval[1]), (0.9, 0.9), label='Intervals', linewidth=5, marker='', color='b')
            first_interval = False
        else:
            ax.plot((interval[0], interval[1]), (0.9, 0.9), linewidth=5, marker='', color='b')

    ax.legend()
    plt.yticks(arange(-0.1, 1.1, 1.2))
    plt.xticks(arange(0, 1.01, 0.1))
    fig.savefig(os.path.join(dir_path, 'a.png'))
    fig.clf()


def c_part(dir_path, distribution):
    k = 2
    T = 100
    ms = [10 + 5*i for i in range(19)]
    true_errors = []
    empirical_errors = []
    for m in ms:
        sum_true_errors = 0
        sum_empirical_errors = 0
        for i in range(T):
            xs, ys = create_training_set(m, distribution)
            intervals, besterror = find_best_interval(xs, ys, k)
            sum_true_errors += true_error(intervals, distribution)
            sum_empirical_errors += float(besterror)/m
        true_errors.append(sum_true_errors/T)
        empirical_errors.append(sum_empirical_errors/T)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ms, true_errors, '.r-', label='True error')
    ax.plot(ms, empirical_errors, '.b-', label='Empirical error')
    plt.xticks(arange(10, 101, 10))
    plt.xlabel('Training set size', fontsize=18)
    plt.ylabel('Error', fontsize=16)
    plt.legend()
    fig.savefig(os.path.join(dir_path, 'c.png'))
    fig.clf()


def d_part(dir_path, distribution):
    m = 50
    xs, ys = create_training_set(m, distribution)

    ks = [i for i in range(1, 21)]
    true_errors = []
    empirical_errors = []
    for k in ks:
        intervals, besterror = find_best_interval(xs, ys, k)
        true_errors.append(true_error(intervals, distribution))
        empirical_errors.append(float(besterror) / m)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ks, true_errors, '.r-', label='True error')
    ax.plot(ks, empirical_errors, '.b-', label='Empirical error')
    plt.legend()
    plt.xlabel('k', fontsize=18)
    plt.ylabel('Error', fontsize=16)
    plt.xticks(arange(1, 20.5, 1))
    fig.savefig(os.path.join(dir_path, 'd.png'))
    fig.clf()


def e_part(dir_path, distribution):
    m = 50
    ks = [i for i in range(1, 21)]
    xs, ys = create_training_set(m, distribution)
    hypothesises = []
    training_scores = []
    for k in ks:
        intervals, besterror = find_best_interval(xs, ys, k)
        hypothesises.append([k, intervals])
        training_scores.append(float(besterror)/m)
    xs, ys = create_training_set(m, distribution)
    best_fitting = empirical_error(xs, ys, hypothesises[0][1])
    best_k = hypothesises[0][0]
    holdout_scores = [best_fitting]
    for hypothesis in hypothesises[1:]:
        score = empirical_error(xs, ys, hypothesis[1])
        holdout_scores.append(score)
        if score < best_fitting:
            best_fitting = score
            best_k = hypothesis[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ks, training_scores, '.r-', label='Training set')
    ax.plot(ks, holdout_scores, '.b-', label='Holdout set')
    plt.legend()
    plt.xlabel('k', fontsize=18)
    plt.ylabel('Error', fontsize=16)
    plt.xticks(arange(1, 20.5, 1))
    fig.savefig(os.path.join(dir_path, 'e.png'))
    fig.clf()
    print "Holdout validation test found that the smallest error is when k=%d" % best_k


dir_path = repr(os.path.dirname(os.path.realpath(sys.argv[0]))).strip("'")

distribution = [((0, 0.25), 0.8),
                ((0.25, 0.5), 0.1),
                ((0.5, 0.75), 0.8),
                ((0.75, 1), 0.1)]

if len(sys.argv) < 2:
    print "Please enter which part do you want to execute - a,c,d,e or all"
    exit()
cmds = sys.argv[1:]
for cmd in cmds:
    if cmd not in ['a', 'c', 'd', 'e', 'all']:
        print "Unknown argument %s. please run with a, c, d, e or all" % cmd
        exit()

if 'a' in cmds or 'all' in cmds:
    a_part(dir_path, distribution)
if 'c' in cmds or 'all' in cmds:
    c_part(dir_path, distribution)
if 'd' in cmds or 'all' in cmds:
    d_part(dir_path, distribution)
if 'e' in cmds or 'all' in cmds:
    e_part(dir_path, distribution)
