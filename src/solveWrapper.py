# Useful for multiprocessing, required to be in its own class
def solveWrapper(tupNode):

    sokopathM = SokoSolver("manhattan",0)

    spathM = sokopathM.astar(tupNode[0], tupNode[1])
    return (sokopathM.nExplored, len(list(spathM)))

