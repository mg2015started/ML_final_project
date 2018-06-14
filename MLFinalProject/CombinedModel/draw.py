from tools import  *
import matplotlib.pyplot as plt



batch_l = [0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200, 7800,8400, 9000]
test_acc_l = [0.10259999709203839, 0.29949998974800107, 0.49379998713731765,
              0.738299983739853, 0.837499989271164, 0.8777000033855438,
              0.8961000120639802, 0.9066000139713287, 0.9129000145196915,
              0.9240000122785568, 0.9302000135183335, 0.9336000156402587,
              0.9381000143289566, 0.9364000123739242, 0.9431000167131424,
              0.9457000106573105]
svm_acc_l = [0.7088, 0.8599, 0.9328, 0.948, 0.9538, 0.9579]

bl2 = []
for i in range(len(batch_l)):
    if i % 3 == 0:
        bl2.append(batch_l[i])

fig = plt.figure()
plt.plot(batch_l, test_acc_l, 'o-', label="CNN acc")
plt.plot(bl2, svm_acc_l, 'o-', label="CNNSVM acc")
plt.xlabel("batch num")
plt.ylabel("acc")
plt.legend(loc='bottom right')
plt.title("CNN and CNNSVM acc")
plt.show()


path = ["conv.dat"]

for p in path:
    bunch = readbunchobj(p)
    batch_l = bunch.batch_l
    train_loss_l = bunch.train_loss_l
    train_acc_l = bunch.train_acc_l
    test_acc_l = bunch.test_acc_l
    svm_acc_l = bunch.svm_acc_l

    print(max(test_acc_l))
    print(max(svm_acc_l))
    bl2 = []
    for i in range(len(batch_l)):
        if i%3==0:
            bl2.append(batch_l[i])

    fig = plt.figure()
    plt.plot(batch_l, test_acc_l, 'o-', label="CNN acc")
    plt.plot(bl2, svm_acc_l, 'o-', label="CNNSVM acc")
    plt.xlabel("batch num")
    plt.ylabel("acc")
    plt.legend(loc='bottom right')
    plt.title("CNN and CNNSVM acc")
    plt.show()

#0.9564
#0.9656
