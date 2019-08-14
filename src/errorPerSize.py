# Error per size graph, uses the output error produced by loadnetError.py
import matplotlib
import matplotlib.pyplot as plt 

e1err9 = "errpersolBest1EpochWin1-med9.out"
e1err12 = "errpersolBest1EpochWin1-med12.out"
e1err15 = "errpersolBest1EpochWin1-med15.out"
e1err18 = "errpersolBest1EpochWin1-med18.out"

e50err9 = "errpersolBest1Epoch50Win1-med9.out"
e50err12 = "errpersolBest1Epoch50Win1-med12.out"
e50err15 = "errpersolBest1Epoch50Win1-med15.out"
e50err18 = "errpersolBest1Epoch50Win1-med18.out"

e150err9 = "errpersolBest1Epoch150Win1-med9.out"
e150err12 = "errpersolBest1Epoch150Win1-med12.out"
e150err15 = "errpersolBest1Epoch150Win1-med15.out"
e150err18 = "errpersolBest1Epoch150Win1-med18.out"

e100err9 = "errpersolBest1Epoch100Win1-09t-med9.out"
e100err12 = "errpersolBest1Epoch100Win1-09t-med12.out"
e100err15 = "errpersolBest1Epoch100Win1-09t-med15.out"
e100err18 = "errpersolBest1Epoch100Win1-09t-med18.out"

e100errPr9 = "prbestEPS-E100Win1-09t-med9.out"
e100errPr12 = "prbestEPS-E100Win1-09t-med12.out"
e100errPr15 = "prbestEPS-E100Win1-09t-med15.out"
e100errPr18 = "prbestEPS-E100Win1-09t-med18.out"

e2errCheb9 = "gcncheb4skip-med9.out"
e2errCheb12 = "gcncheb4skip-med12.out"
e2errCheb15 = "gcncheb4skip-med15.out"
e2errCheb18 = "gcncheb4skip-med18.out"

epocherr = ["Epoch 1","Epoch 50", "Epoch 150", "Epoch 09t 100", "Epoch 09t 100 d15", "Epoch 2 Cheb"]
plotlist = ["Medium9","Medium12","Medium15","Medium18"]

txtlist = [e1err9, e1err12, e1err15, e1err18, e50err9, e50err12, e50err15, e50err18, e150err9, e150err12, e150err15, e150err18, e100err9, e100err12, e100err15, e100err18, e100errPr9, e100errPr12, e100errPr15, e100errPr18, e2errCheb9, e2errCheb12, e2errCheb15, e2errCheb18]
rese1 = []
rese50 = []
rese150 = []
rese100 = []
rese100d15 = []
rese2cheb = []
for idx, fname in enumerate(txtlist):
    with open(fname, 'r') as f:
        lines = f.read().splitlines()
        if idx < 4:
            rese1.append(float(lines[-2]))
        elif idx >= 4 and idx < 8:
            rese50.append(float(lines[-2]))
        elif idx >= 8 and idx < 12:
            rese150.append(float(lines[-2]))
        elif idx >= 12 and idx < 16:
            rese100.append(float(lines[-2]))
        elif idx >= 16 and idx < 20:
            rese100d15.append(float(lines[-2]))
        else:
            rese2cheb.append(float(lines[-2]))

xt = ["9","12","15","18"]
x = range(4)
plt.figure()
plt.plot(x,rese1,label=epocherr[0])
plt.plot(x,rese50,label=epocherr[1])
plt.plot(x,rese150,label=epocherr[2])
plt.plot(x,rese100,label=epocherr[3])
plt.plot(x,rese100d15,label=epocherr[4])
plt.plot(x,rese2cheb,label=epocherr[5])

plt.title('CNNReLU Win1 and GCN Cheb')
plt.xlabel('Level Size')
plt.ylabel('L1 loss')
plt.xticks(x,xt)
plt.legend()
#ax1.bar(ErrPerSolLength.keys(), ErrPerSolLength.values(), width=0.8)
#ax1.set_title('Predictions')
#ax2.bar(targets.keys(), targets.values(), width=0.8)
#ax2.set_title('Targets')
plt.tight_layout()
plt.savefig("LevelsizeErrorCheb.png")
plt.savefig("LevelsizeErrorCheb.svg")





