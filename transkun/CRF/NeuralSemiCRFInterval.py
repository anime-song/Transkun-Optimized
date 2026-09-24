import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List,Tuple, Optional

"""
    The neural semiCRF module for handling multiple tracks of non-overlapping intervalsc
    Author: Yujia Yan
"""

     
@torch.jit.script
def _viterbi_backward_tensors(
    score: torch.Tensor,
    noise_score: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Viterbi DPをデバイス上で実行し、tracebackに必要な状態を返す。"""
    assert score.dim() == 3 and score.shape[0] == score.shape[1]
    time_steps = int(score.shape[0])
    track_count = int(score.shape[2])

    # The original decoder accumulated in the default torch float dtype.
    q = torch.zeros(time_steps, track_count, device=score.device)
    pointers = torch.jit.annotate(List[torch.Tensor], [])
    q[time_steps - 1] = score[time_steps - 1, time_steps - 1, :] * (
        score[time_steps - 1, time_steps - 1, :] > 0
    )

    for offset in range(1, time_steps):
        begin = time_steps - offset - 1

        # 区間候補とskip候補を別々に評価し、時刻ごとの一時的なcatを作らない。
        interval_values = q[begin + 1 :, :] + score[begin + 1 :, begin, :]
        best_interval_value, interval_selection = interval_values.max(dim=0)
        skip_value = q[begin + 1, :] + noise_score[begin, :]

        # 旧実装ではskipが候補の先頭にあり、同点時はskipが選ばれていた。
        use_interval = best_interval_value > skip_value
        pointers.append(torch.where(use_interval, interval_selection, -1))
        best_value = torch.where(use_interval, best_interval_value, skip_value)

        singleton_mask = score[begin, begin, :] > 0
        q[begin] = best_value + score[begin, begin, :] * singleton_mask

    # traceback順のpointerをtrack-majorで直接積み、転置コピーを発生させない。
    return (
        torch.stack(pointers, dim=1),
        torch.diagonal(score, dim1=0, dim2=1) > 0,
    )


def viterbiBackward(
    score: torch.Tensor,
    noiseScore: torch.Tensor,
    forcedStartPos: Optional[List[int]] = None,
    *,
    backend: str = "auto",
) -> List[List[Tuple[int, int]]]:
    """
    Decode the best sequence of closed intervals.

    Args:
        score: [T, T, B], indexed by [end, begin, track].
        noiseScore: [T-1, B] skip scores.
        forcedStartPos: Optional start position for each track.
        backend: auto selects Triton for CUDA float32 if installed.
    """
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of {'auto', 'torch', 'triton'}")
    use_triton = backend == "triton" or (
        backend == "auto" and score.device.type == "cuda"
        and score.dtype == torch.float32 and noiseScore.dtype == torch.float32
        and score.shape[0] >= 2
    )
    if use_triton:
        if score.device.type != "cuda":
            raise ValueError("Triton Semi-CRF decoding requires a CUDA tensor")
        try:
            from .semi_crf_triton import viterbi_backward_triton
        except ModuleNotFoundError as exc:
            if exc.name is None or not exc.name.startswith("triton"):
                raise
            if backend == "triton":
                raise RuntimeError(
                    "Triton Semi-CRF decoding requires the triton package"
                ) from exc
            use_triton = False
        if use_triton:
            return viterbi_backward_triton(score, noiseScore, forcedStartPos)

    if score.shape[0] == 1:
        diagonal = (score[0, 0] > 0).cpu().tolist()
        return [[(0, 0)] if selected else [] for selected in diagonal]

    pointers, diag_inclusion = _viterbi_backward_tensors(score, noiseScore)
    time_steps = int(score.shape[0])
    track_count = int(score.shape[2])

    # Copy only tracks that can contain notes; traverse Python lists rather
    # than repeatedly reading scalar values from CPU tensors.
    active_tracks = (pointers >= 0).any(dim=1) | diag_inclusion.any(dim=1)
    active_index = active_tracks.nonzero().flatten()
    pointer_values = pointers.index_select(0, active_index).cpu().tolist()
    diag_values = diag_inclusion.index_select(0, active_index).cpu().tolist()

    if forcedStartPos is None:
        forcedStartPos = [0] * track_count

    result: List[List[Tuple[int, int]]] = [[] for _ in range(track_count)]
    for row_index, track in enumerate(active_index.tolist()):
        position = forcedStartPos[track]
        track_result: List[Tuple[int, int]] = []
        pointer_row = pointer_values[row_index]
        current_diag = diag_values[row_index]

        while position < time_steps - 1:
            selection = int(pointer_row[time_steps - position - 2])

            if bool(current_diag[position]):
                track_result.append((position, position))

            if selection < 0:
                position += 1
            else:
                end = selection + position + 1
                track_result.append((position, end))
                position = end

        if bool(current_diag[time_steps - 1]):
            track_result.append((time_steps - 1, time_steps - 1))

        result[track] = track_result

    return result


@torch.jit.script
def viterbi(score, noiseScore, forcedStartPos: Optional[List[int]]=None):
    # score: [nEndPos,  nBeginPos, nBatch]
    # noiseScore: [nEndPos-1, nBatch]

    assert(len(score.shape) == 3)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]
    
    v = torch.zeros(T, nBatch, device = score.device)

    # for back tracking
    ptr = []


    v[0] = score[0,0, :]* (score[0,0,:]>0)

    for i in range(1,T):

        # update v_on
        subScore = score[i, :i, :]

        tmp = torch.cat(
            [
            v[i-1:i,:] + noiseScore[i-1, :],              # skip
            v[:i, :]+ subScore       # an interval 
            ],
            dim = 0
        )
        
        curV, selection = tmp.max(dim = 0)

        ptr.append(selection-1)


        singletonMask = score[i, i,:]>0

        v[i] = curV+ score[i,i,:]*singletonMask


    vFinal = v[-1]
    # print(vFinal)

    ptr= torch.stack(ptr, dim = 0).cpu()

    scoreDiagInclusion = (torch.diagonal(score, dim1= 0, dim2=1)>0).cpu()




    # perform backtracking 
    result: List[List[Tuple[int, int]]]  =  []
    # print(ptr)

    if forcedStartPos is None:
        forcedStartPos = [T-1]* nBatch
    
    for idx in range(nBatch):
        j = forcedStartPos[idx]
        # c = int(cFinal[idx] )

        curResult : List[Tuple[int, int]]  = []

        curDiag = scoreDiagInclusion[idx]

        while j>0:
            # print(j)
            # if c == 0:
            curSelecton= int(ptr[j-1][idx])

            if bool(curDiag[j]):
                curResult.append((j,j))


            if curSelecton<0:
                # skip
                j = j-1
            else:
                i =curSelecton 

                # print((i,j))
                curResult.append((i,j))

                j = i
        
        if score[0,0, idx]>0:
            curResult.append((0,0))
        
        # # flip the list
        curResult.reverse()


        result.append(curResult)

    
    return result



@torch.jit.script
def computeLogZ(score, noiseScore):
    # [nEndPos,  nBeginPos, nBatch]
    assert(len(score.shape) == 3)
    assert(len(noiseScore.shape) == 2)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]

    assert(noiseScore.shape[0] == T-1)


    v = F.softplus(score[0,0,:]).unsqueeze(0)

    for i in range(1, T):
        subScore = score[i, :i, :]


        tmp = torch.cat(
            [
            v[i-1:i,:] + noiseScore[i-1,:],              # skip
            v[:i, :]+ subScore       # an interval 
            ],
            dim = 0
        )

        curV= tmp.logsumexp(dim = 0) + F.softplus(score[i, i,:])

        v = torch.cat( [ v, curV.unsqueeze(0)], dim = 0)
        # curV= tmp.logsumexp(dim = 0) + singleScoreSP[i,:]#F.softplus(score[i, i,:])
        # curV = torch.logaddexp(
                # v[i-1,:] + noiseScore[i-1,:],              # skip
                # torch.logsumexp(v[:i, :]+subScore, dim = 0) )
        # curV = curV+singleScoreSP[i,:]
        
        # v = torch.cat( [ v, curV.unsqueeze(0)], dim = 0)
    
    vFinal = v[-1]
    

    return vFinal

# the autodiff backward is too slow, let write our own
@torch.jit.script
def forward_backwardOld(score, noiseScore):
    # [nEndPos,  nBeginPos, nBatch]
    assert(len(score.shape) == 3)
    assert(len(noiseScore.shape) == 2)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]
    assert(noiseScore.shape[0] == T-1)


    # forward pass
    singleScoreSP = F.softplus(torch.diagonal(score, dim1=0, dim2=1)).transpose(-1,-2)
    # v = F.softplus(score[0,0,:]).unsqueeze(0)

    v = score.new_zeros(T, nBatch)
    v[0] = singleScoreSP[0, :]
    # v = singleScoreSP[0, :].unsqueeze(0)

    for i in range(1, T):
        subScore = score[i, :i, :]


        # tmp = torch.cat(
            # [
            # v[i-1:i,:] + noiseScore[i-1,:],              # skip
            # v[:i, :]+ subScore       # an interval 
            # ],
            # dim = 0
        # )

        # curV= tmp.logsumexp(dim = 0) + singleScoreSP[i,:]#F.softplus(score[i, i,:])
        # curV = torch.logaddexp(
                # v[i-1,:] + noiseScore[i-1,:],              # skip
                # torch.logsumexp(v[:i, :]+subScore, dim = 0) )
        # curV = curV+singleScoreSP[i,:]
        
        # v = torch.cat( [ v, curV.unsqueeze(0)], dim = 0)

        v[i] = torch.logaddexp(
                v[i-1,:] + noiseScore[i-1,:],              # skip
                torch.logsumexp(v[:i, :]+subScore, dim = 0) )
        v[i] += singleScoreSP[i,:]
        # v[i] = curV

    logZ = v[-1]



    # then do a backward pass
    scoreT = score.transpose(0,1).contiguous()
    # print("backward")

    # q = F.softplus(score[T-1, T-1, :]).unsqueeze(0)
    q = score.new_zeros(T, nBatch)
    q[T-1] = singleScoreSP[T-1, :]
    # q = singleScoreSP[T-1, :].unsqueeze(0)
    for i in range(1, T):
        subScore  = scoreT[T-i-1, T-i:,:]
        # tmp = torch.cat(
            # [
            # q[0:1, :] + noiseScore[T-i-1,:],              # skip
            # q + subScore       # an interval 
            # ],
            # dim = 0
        # )
        # curQ = tmp.logsumexp(dim =0) + singleScoreSP[T-i-1,:]#F.softplus(scoreT[T-i-1, T-i-1, :])
        # curQ= torch.logaddexp(
            # q[0, :] + noiseScore[T-i-1,:],              # skip
            # torch.logsumexp(q + subScore, dim=0)       # an interval 
            # )
        # curQ = curQ + singleScoreSP[T-i-1,:]

        # q = torch.cat([curQ.unsqueeze(0), q], dim = 0)

        q[T-i-1] = torch.logaddexp(
            q[T-i, :] + noiseScore[T-i-1,:],              # skip
            torch.logsumexp(q[T-i:] + subScore, dim=0)       # an interval 
            ) + singleScoreSP[T-i-1,:]

        # q = torch.cat([curQ.unsqueeze(0), q], dim = 0)


    # print("v:", v[-1])
    # print("q", q[0])
    # print(v[-1]-q[0])
    # grad except for the diagonal can be computed in the following way
    # grad = (v.unsqueeze(0)+ q.unsqueeze(1) + score -logZ ).exp()

    # diagonal elements are computed in this way
    # grad = (v.unsqueeze(0)+ q.unsqueeze(1) + score - 2* F.softplus(score) -logZ ).exp()

    grad = (v.unsqueeze(0)+ q.unsqueeze(1)-logZ + score)#.exp()

    # minus 2* F.softplus(diag of score)
    grad = grad - 2*torch.diag_embed(F.softplus(torch.diagonal(score, dim1 = 0, dim2 = 1)), dim1= 0, dim2 = 1)

    # grad = grad#*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)


     
    grad = grad*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    grad = grad.exp()

    grad = grad*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    # for grad of noise, it is similar

    
    gradNoise = (v[:-1]+ q[1:] + noiseScore-logZ)
    # gradNoise = gradNoise - gradNoise*(gradNoise>0)
    gradNoise = gradNoise.exp()

    
    # zeroout all the upper triangular part
    # print(grad)
    # print(grad[:,:,0])
    # print("grad:", grad[:,:,0])
    # print("gradNoise:",gradNoise[:,0])


    return logZ, grad, gradNoise


@torch.jit.script
def forward_backward(score, noiseScore):
    # [nEndPos,  nBeginPos, nBatch]
    assert(len(score.shape) == 3)
    assert(len(noiseScore.shape) == 2)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]
    assert(noiseScore.shape[0] == T-1)
    

    # make score symmetric
    scoreFlip = torch.flip(score, dims = [0,1]).transpose(0,1)
    noiseScoreFlip = torch.flip(noiseScore, (0,))


    scoreFB= torch.cat([score, scoreFlip], dim = -1)
    noiseScoreFB = torch.cat([noiseScore, noiseScoreFlip], dim = -1)

    
    # forward pass
    singleScoreSP = F.softplus(torch.diagonal(scoreFB, dim1=0, dim2=1)).transpose(-1,-2)
    # v = F.softplus(score[0,0,:]).unsqueeze(0)

    v = score.new_zeros(T, nBatch*2)
    v[0] = singleScoreSP[0, :]
    # v = singleScoreSP[0, :].unsqueeze(0)

    for i in range(1, T):
        subScore = scoreFB[i, :i, :]



        v[i] = torch.logaddexp(
                v[i-1,:] + noiseScoreFB[i-1,:],              # skip
                torch.logsumexp(v[:i, :]+subScore, dim = 0) )
        v[i] += singleScoreSP[i,:]
        # v[i] = curV
    v,q = torch.chunk(v, 2, dim = -1)

    q = torch.flip(q, (0,))


    logZ = v[-1]


    # print(logZ)
    scoreFB = None
    noiseScoreFB = None

    grad = (v.unsqueeze(0)+ ((q.unsqueeze(1)-logZ) + score))#.exp()

    # minus 2* F.softplus(diag of score)
    grad = grad - 2*torch.diag_embed(F.softplus(torch.diagonal(score, dim1 = 0, dim2 = 1)), dim1= 0, dim2 = 1)

    # grad = grad#*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    # do some clamping for satbility 
    # grad = 
    # grad = grad - grad*(grad>0)

     
    grad = grad*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    grad = grad.exp()

    grad = grad*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    # for grad of noise, it is similar

    
    gradNoise = (v[:-1]+ q[1:] + noiseScore-logZ)
    # gradNoise = gradNoise - gradNoise*(gradNoise>0)
    gradNoise = gradNoise.exp()

    
    # print(grad)
    # print(grad[:,:,0])
    # print("grad:", grad[:,:,0])
    # print("gradNoise:",gradNoise[:,0])


    return logZ, grad, gradNoise


class ComputeLogZFasterGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, score, noiseScore):
        logz, grad, gradNoise = forward_backward(score, noiseScore)
        ctx.save_for_backward(grad, gradNoise)

        return logz

    @staticmethod
    def backward(ctx, grad_output):
        grad, gradNoise, = ctx.saved_tensors
        assert(grad_output.shape[-1] == grad.shape[-1])
        return grad*grad_output, gradNoise*grad_output
        

computeLogZFasterGrad = ComputeLogZFasterGrad.apply


@torch.jit.script
def evalPathSlow(intervals: List[List[Tuple[int, int]]], score, noiseScore):
    # the naive version
    # print(intervals)
    result = []




    noiseScore = F.pad(noiseScore, (0,0,1,0))
    noiseScoreCum = torch.cumsum(noiseScore, dim = 0)

    for idx, curList in enumerate(intervals):
        v = noiseScoreCum[-1, idx]

        for i, j in curList:
            v = v+ (score[j,i, idx]) - noiseScoreCum[j, idx] + noiseScoreCum[i, idx]
        result.append(v)

    result = torch.stack(result, dim = -1)




    return result
    


    

def evalPath(intervals: List[List[Tuple[int, int]]], score, noiseScore):
    # score: [endpos,  beginPos, nBatch], note that the interval is close
    assert(len(score.shape) == 3)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]

    # first gather then scatter_add
    # idxAll = []
    # for idx, curList in enumerate(intervals):
        # idxAll.append(idx)
        # (score[j,i, idx])

        # result.append(v)
    device = score.device

    noiseScore = F.pad(noiseScore, (0,0,1,0))
    noiseScoreCum = torch.cumsum(noiseScore, dim = 0)

    indices = [ idx+ i*nBatch + j*nBatch*T for idx, curList in enumerate(intervals) for i,j in curList]  
    batchIndices = [ idx for idx, curList in enumerate(intervals) for _ in curList]  



    indices_neg = [idx+ i*nBatch  for idx, curList in enumerate(intervals) for i,j in curList]  
    indices_pos= [ idx+ j*nBatch for idx, curList in enumerate(intervals) for i,j in curList]  


    indices = torch.tensor(indices, device = device, dtype = torch.long)
    batchIndices= torch.tensor(batchIndices, device = device, dtype = torch.long)
    indices_pos = torch.tensor(indices_pos, device = device, dtype = torch.long)
    indices_neg = torch.tensor(indices_neg, device = device, dtype = torch.long)
    gathered_pos = noiseScoreCum.view(-1).gather(0, indices_pos)
    gathered_neg= noiseScoreCum.view(-1).gather(0, indices_neg)

    gathered = score.view(-1).gather(0, indices) - (gathered_pos-gathered_neg)

    result = gathered.new_zeros(nBatch, device = device)

    result = result.scatter_add(-1, batchIndices, gathered)
    result = result+ noiseScoreCum[-1,:]

    return result


class NeuralSemiCRFInterval:
    def __init__(self, score, noiseScore):
        """ The output layer for multiple tracks of non-overlapping intervals

        arguments:
        score -- the score matrix for all possible [begin, end] pairs, which has a shape of [T, T, nBatch],
                 where T is the length of the sequence and nBatch is how many event tracks to be decoded simultaneously.
        noiseScore -- The non-event score for the internval [t, t+1], which has a shape of [T-1, nBatch]
        """
        
        self.score = score
        self.noiseScore = noiseScore
    

    def decode(self, forcedStartPos=None, forward=False, *, backend="auto"):
        if forward:
            if backend not in ("auto", "torch"):
                raise ValueError("forward decoding only supports the torch backend")
            return viterbi(self.score, self.noiseScore, forcedStartPos)
        else:
            return viterbiBackward(self.score, self.noiseScore, forcedStartPos, backend=backend)


    def evalPath(self, intervals):
        """ compute the unnormalized score """
        pathScore = evalPath(intervals, self.score, self.noiseScore)
        return pathScore
            

    def computeLogZ(self, noBackward=False, *, backend="auto"):
        """ compute the log normalization factor """
        if backend not in ("auto", "torch", "triton"):
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        if noBackward:
            if backend == "triton":
                raise ValueError("noBackward log-partition only supports torch")
            return computeLogZ(self.score, self.noiseScore)
        use_triton = backend == "triton" or (
            backend == "auto" and self.score.device.type == "cuda"
            and self.score.dtype == torch.float32
            and self.noiseScore.dtype == torch.float32
        )
        if use_triton:
            if self.score.device.type != "cuda":
                raise ValueError("Triton Semi-CRF loss requires a CUDA tensor")
            try:
                from .semi_crf_loss_triton import compute_log_z_triton
            except ModuleNotFoundError as exc:
                if exc.name is None or not exc.name.startswith("triton"):
                    raise
                if backend == "triton":
                    raise RuntimeError("Triton Semi-CRF loss requires the triton package") from exc
            else:
                return compute_log_z_triton(self.score, self.noiseScore)
        return computeLogZFasterGrad(self.score, self.noiseScore)

    def logProb(self, intervals, noBackward=False, *, backend="auto"):
        return self.evalPath(intervals) - self.computeLogZ(noBackward=noBackward, backend=backend)


if __name__ == "__main__":
    # a simple example
    
    score = ((torch.randn(200,  200, 4))).cuda()
    noiseScore= ((torch.randn(199,  4))).cuda()

    noiseScore.requires_grad_()
    score.requires_grad_()
    optimizer = torch.optim.Adam([score, noiseScore], 1e-3)
    intervals = [
            [(0,2), (4,6),(6,6), (7,8)],
            [(1,2), (3,5), (19,19)],
            [(0,0),(4,7)],
            [],
            ]
    

    for i in range(10000):
        optimizer.zero_grad()
        crf = NeuralSemiCRFInterval(score, noiseScore)
        logP  = crf.evalPath(intervals) - crf.computeLogZ()
        loss = -logP.sum()
        (loss).backward()
        optimizer.step()

        print(loss)

        if i%100==0:
            with torch.no_grad():
                recons = crf.decode()
                print(recons)
