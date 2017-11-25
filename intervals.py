from numpy import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

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
    E[:m+1,0] = cy
    
    # The minimal error of j intervals on 0 points - always 0. No update needed.        
        
    # Fill middle
    for i in range(1, m+1):
        for j in range(1, k+1):
            # The minimal error of j intervals on the first i points:
            
            # Exhaust all the options for the last interval. Each interval boundary is marked as either
            # 0 (Before first point), 1 (after first point, before second), ..., m (after last point)
            options = []
            for l in range(0,i+1):  
                next_errors = E[l,j-1] + (cy[i]-cy[l]) + concatenate([[0], cumsum((-1)**(ys[arange(l, i)] == 1))])
                min_error = argmin(next_errors)
                options.append((next_errors[min_error], (l, arange(l,i+1)[min_error])))

            E[i,j], P[i][j] = min(options)
    
    # Extract best interval set and its error count
    best = []
    cur = P[m][k]
    for i in range(k,0,-1):
        best.append(cur)
        cur = P[cur[0]][i-1]       
        if cur == None:
            break 
    best = sorted(best)
    besterror = E[m,k]
    
    # Convert interval boundaries to numbers in [0,1]
    exs = concatenate([[0], xs, [1]])
    representatives = (exs[1:]+exs[:-1]) / 2.0
    intervals = [(representatives[l], representatives[u]) for l,u in best]

    return intervals, besterror

def create_training_set(m):
    training_set = []
    for i in range(m):
        x = random.uniform(0,1)
        if 0<=x<=0.25 or 0.5<=x<0.75:
            y = random.binomial(1,0.8)
        else:
            y = random.binomial(1, 0.1)
        training_set.append((x,y))
    training_set.sort(key=lambda x:x[0])
    xs = [x[0] for x in training_set]
    ys = [y[1] for y in training_set]
    return xs, ys

def emprical_error(xs, ys, intervals):
    errors = 0
    for i in range(len(xs)):
        if len([True for interval in intervals if interval[0] <= xs[i] <= interval[1]]) > 0:
            if ys[i] == 0:
                errors += 1
        elif ys[i] == 1:
            errors += 1
    return float(errors) / len(xs)


def true_error(intervals):
    def loss_in(interval, d):
        interval_loss = 0.2 if d == 1 else 0.8
        return (interval[1] - interval[0]) * interval_loss

    def loss_out(interval, d):
        interval_loss = 0.9 if d == 1 else 0.1
        return (interval[1] - interval[0]) * interval_loss

    intervals.sort(key=lambda x:x[0])
    current_point = 0
    space_partition = []
    for interval in intervals:
        if interval[0] == current_point:
            space_partition.append([[interval[0],interval[1]],1])
            current_point = interval[1]
        else:
            space_partition.append([[current_point,interval[0]], 0])
            space_partition.append([[interval[0],interval[1]], 1])
            current_point = interval[1]
    loss = 0
    for interval in space_partition:
        if interval[0][1] <= 0.25:
            loss += loss_in(interval[0],interval[1])
            continue
        elif interval[0][0] < 0.25:
            loss += loss_in((interval[0][0],0.25),interval[1])
            interval[0][0] = 0.25

        if interval[0][1] <= 0.5:
            loss += loss_out(interval[0],interval[1])
            continue
        elif interval[0][0] < 0.5:
            loss += loss_out((interval[0][0],0.5),interval[1])
            interval[0][0] = 0.5

        if interval[0][1] <= 0.75:
            loss += loss_in(interval[0],interval[1])
            continue
        elif interval[0][0] < 0.75:
            loss += loss_in((interval[0][0],0.75), interval[1])
            interval[0][0] = 0.75

        loss += loss_out(interval[0],interval[1])

    return loss

def a_part():
    lines = [0.25,0.5,0.75]
    k = 2
    xs, ys = create_training_set(100)
    intervals, besterror = find_best_interval(xs,ys,k)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs,ys, s=2)
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
    fig.savefig("a.png")
    fig.clf()

def c_part():
    k=2
    ms = [10 + 5*i for i in range(19)]
    true_errors = []
    empirical_errors = []
    for m in ms:
        temp_true = []
        temp_empirical = []
        for i in range(100):
            xs, ys = create_training_set(m)
            intervals, besterror = find_best_interval(xs, ys, k)
            temp_true.append(true_error(intervals))
            temp_empirical.append(float(besterror)/m)
        true_errors.append(sum(temp_true)/100)
        empirical_errors.append(sum(temp_empirical)/100)
    plt.plot(ms,true_errors, '.r-', label='True error')
    plt.plot(ms,empirical_errors, '.b-', label='Empirical error')
    plt.xticks(arange(10, 101, 10))
    plt.xlabel('Training set size', fontsize=18)
    plt.ylabel('Error', fontsize=16)
    plt.legend()
    plt.savefig("c.png")
    plt.clf()

def d_part():
    m=50
    xs, ys = create_training_set(m)

    ks = [i for i in range(1,21)]
    true_errors = []
    empirical_errors = []
    for k in ks:
        intervals, besterror = find_best_interval(xs, ys, k)
        true_errors.append(true_error(intervals))
        empirical_errors.append(float(besterror) / m)

    plt.plot(ks, true_errors, '.r-', label='True error')
    plt.plot(ks, empirical_errors, '.b-', label='Empirical error')
    plt.legend()
    plt.xlabel('k', fontsize=18)
    plt.ylabel('Error', fontsize=16)
    plt.xticks(arange(1, 20.5, 1))
    plt.savefig("d.png")
    plt.clf()

def e_part():
    m = 50
    ks = [i for i in range(1, 21)]
    xs, ys = create_training_set(m)
    hypothesises = []
    training_scores = []
    for k in ks:
        intervals, besterror = find_best_interval(xs, ys, k)
        hypothesises.append([k,intervals])
        training_scores.append(float(besterror)/m)
    xs, ys = create_training_set(m)
    best_fitting = emprical_error(xs, ys,hypothesises[0][1])
    best_k = hypothesises[0][0]
    holdout_scores = [best_fitting]
    for hypothesis in hypothesises[1:]:
        score = emprical_error(xs, ys,hypothesis[1])
        holdout_scores.append(score)
        if score < best_fitting:
            best_fitting = score
            best_k = hypothesis[0]

    plt.plot(ks, training_scores, '.r-', label='Training set')
    plt.plot(ks, holdout_scores, '.b-', label='Holdout set')
    plt.legend()
    plt.xlabel('K', fontsize=18)
    plt.ylabel('Error', fontsize=16)
    plt.xticks(arange(1, 20.5, 1))
    plt.savefig("e.png")
    plt.clf()
    print best_k

if len(sys.argv) < 2:
    print "please enter which part do you want to execute - a,c,d,e or all"
    exit()
cmds = sys.argv[1:]
for cmd in cmds:
    if cmd not in ['a','c','d','e','all']:
        print "unkown argument %s. please run with a, c, d, e or all" % cmd
        exit()

if 'a' in cmds or 'all' in cmds:
    a_part()
if 'c' in cmds or 'all' in cmds:
    c_part()
if 'd' in cmds or 'all' in cmds:
    d_part()
if 'e' in cmds or 'all' in cmds:
    e_part()
