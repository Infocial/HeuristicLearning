# Produces error per solution graphs from numpy dicts created by loadnetError.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


e1err9 = np.load("errpersolBest1EpochWin1-med9.npy",allow_pickle=True).item()
e1err12 = np.load("errpersolBest1EpochWin1-med12.npy",allow_pickle=True).item()
e1err15 = np.load("errpersolBest1EpochWin1-med15.npy",allow_pickle=True).item()
e1err18 = np.load("errpersolBest1EpochWin1-med18.npy",allow_pickle=True).item()

e50err9 = np.load("errpersolBest1Epoch50Win1-med9.npy",allow_pickle=True).item()
e50err12 = np.load("errpersolBest1Epoch50Win1-med12.npy",allow_pickle=True).item()
e50err15 = np.load("errpersolBest1Epoch50Win1-med15.npy",allow_pickle=True).item()
e50err18 = np.load("errpersolBest1Epoch50Win1-med18.npy",allow_pickle=True).item()

e150err9 = np.load("errpersolBest1Epoch150Win1-med9.npy",allow_pickle=True).item()
e150err12 = np.load("errpersolBest1Epoch150Win1-med12.npy",allow_pickle=True).item()
e150err15 = np.load("errpersolBest1Epoch150Win1-med15.npy",allow_pickle=True).item()
e150err18 = np.load("errpersolBest1Epoch150Win1-med18.npy",allow_pickle=True).item()

e100err9 = np.load("errpersolBest1Epoch100Win1-09t-med9.npy",allow_pickle=True).item()
e100err12 = np.load("errpersolBest1Epoch100Win1-09t-med12.npy",allow_pickle=True).item()
e100err15 = np.load("errpersolBest1Epoch100Win1-09t-med15.npy",allow_pickle=True).item()
e100err18 = np.load("errpersolBest1Epoch100Win1-09t-med18.npy",allow_pickle=True).item()

e100errPr9 = np.load("prbestEPS-E100Win1-09t-med9.npy",allow_pickle=True).item()
e100errPr12 = np.load("prbestEPS-E100Win1-09t-med12.npy",allow_pickle=True).item()
e100errPr15 = np.load("prbestEPS-E100Win1-09t-med15.npy",allow_pickle=True).item()
e100errPr18 = np.load("prbestEPS-E100Win1-09t-med18.npy",allow_pickle=True).item()

e2errCheb9 = np.load("gcncheb4skip-med9.npy",allow_pickle=True).item()
e2errCheb12 = np.load("gcncheb4skip-med12.npy",allow_pickle=True).item()
e2errCheb15 = np.load("gcncheb4skip-med15.npy",allow_pickle=True).item()
e2errCheb18 = np.load("gcncheb4skip-med18.npy",allow_pickle=True).item()

epocherr = ["Epoch 1 ","Epoch 50 ", "Epoch 150 ", "Epoch 09t 100", "Epoch 09t 100 d15", "Epoch 2 Cheb"]

dictlist = [e1err9, e1err12, e1err15, e1err18, e50err9, e50err12, e50err15, e50err18, e150err9, e150err12, e150err15, e150err18, e100err9, e100err12, e100err15, e100err18, e100errPr9, e100errPr12, e100errPr15, e100errPr18, e2errCheb9, e2errCheb12, e2errCheb15, e2errCheb18]
plotlist = ["Medium9","Medium12","Medium15","Medium18"]

plt.figure()
for idx, err in enumerate(dictlist):

    for key in err:
        err[key] = np.mean(err[key])

    lists = sorted(err.items())
    x, y = zip(*lists)
    if idx < 4:
        #plt.plot(x[:40],y[:40],label=epocherr[0]+plotlist[idx])
        #plt.plot(x[:],y[:],label=epocherr[0]+plotlist[idx])
        print("bla")
    elif idx >= 4 and idx < 8:
        #plt.plot(x[:40],y[:40],label=epocherr[1]+plotlist[idx-4])
        #plt.plot(x[:],y[:],label=epocherr[1]+plotlist[idx-4])
        print("bla")
    elif idx>= 8 and idx < 12:
        #plt.plot(x[:40],y[:40],label=epocherr[2]+plotlist[idx-8])
        #plt.plot(x[:60],y[:60],label=epocherr[2]+plotlist[idx-8])
        print("bla")
    elif idx>= 12 and idx < 16:
        #plt.plot(x[:40],y[:40],label=epocherr[3]+plotlist[idx-12])
        #plt.plot(x[:],y[:],label=epocherr[3]+plotlist[idx-12])
        print("bla")
    elif idx>= 16 and idx < 20:
        #plt.plot(x[:40],y[:40],label=epocherr[4]+plotlist[idx-16])
        plt.plot(x[:],y[:],label=epocherr[4]+plotlist[idx-16])
    else:
        plt.plot(x[:],y[:],label=epocherr[5]+plotlist[idx-20])
#fig, (ax1, ax2) = plt.subplots(2)

plt.title('CNNReLU and GCN Cheb')
plt.xlabel('Solution length')
plt.ylabel('L1 loss')
plt.legend()
#ax1.bar(ErrPerSolLength.keys(), ErrPerSolLength.values(), width=0.8)
#ax1.set_title('Predictions')
#ax2.bar(targets.keys(), targets.values(), width=0.8)
#ax2.set_title('Targets')
plt.tight_layout()
plt.savefig("netcompare-errorpersolution.png")
plt.savefig("netcompare-errorpersolution.svg")
