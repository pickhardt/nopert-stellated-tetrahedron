import sys, time
sys.path.insert(0,'certificates'); sys.path.insert(0,'.')
from multiprocessing import Pool
import rc_dlo
from rc_dlo2 import dlo_window2
from rc_dlo import I
d0,d1,p0,p1,nd,npp,path = float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),sys.argv[7]
de=[d0+(d1-d0)*k/nd for k in range(nd+1)]
pe=[p0+(p1-p0)*k/npp for k in range(npp+1)]
def cell(ij):
    i,j=ij
    try: dl=dlo_window2(I(de[i],de[i+1]),I(pe[j],pe[j+1]))
    except Exception: dl=None
    return (i,j, 0.0 if dl is None else dl)
if __name__=='__main__':
    jobs=[(i,j) for i in range(nd) for j in range(npp)]
    t0=time.time()
    with Pool() as pool: res=pool.map(cell,jobs,chunksize=32)
    grid=[[0.0]*npp for _ in range(nd)]
    for i,j,v in res: grid[i][j]=v
    nz=sum(1 for _,_,v in res if v>0)
    with open(path,'w') as f:
        f.write('%d %d\n'%(nd,npp))
        f.write(' '.join(repr(x) for x in de)+'\n')
        f.write(' '.join(repr(x) for x in pe)+'\n')
        for row in grid: f.write(' '.join(repr(x) for x in row)+'\n')
    print('grid %dx%d, %d/%d certifiable (%.0f%%), %.0fs'%(nd,npp,nz,len(jobs),100*nz/len(jobs),time.time()-t0))
