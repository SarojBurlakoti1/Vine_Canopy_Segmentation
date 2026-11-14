"""Sequence-alignment audit of 2023 capture order (tier C) in split_assignment_final.csv.
Hypothesis: on each session the photographer walked the same route, so aligned positions with
identical species inside long conserved runs are likely the same physical plant.
Output: candidate cross-session matches and how often a test image's candidate counterpart is in train/val.
This is an inference to be verified against field records, NOT a confirmed plant identity."""
import pandas as pd, numpy as np, sys, itertools, json
df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
df['orig'] = df.roboflow_filename.str.split('.rf.').str[0]
ab = {'Akebia':'A','Campsis':'C','Hydrangea':'H','Lonicera':'L','Madame Galen':'M','Parthenocissus':'P','Wisteria':'W'}
c = df[(df.tier=='C_independent_dated') & (~df.orig.str.contains('-2-'))].copy()
c['dt'] = pd.to_datetime(c.date_taken, format='%Y:%m:%d %H:%M:%S'); c=c.sort_values('dt')
c['session'] = c.dt.dt.date.astype(str)
sess = {k:g.reset_index(drop=True) for k,g in c.groupby('session') if len(g)>=50}
def nw(a,b,m=2,mm=-2,gap=-1):
    n,k=len(a),len(b); S=np.zeros((n+1,k+1)); P=np.zeros((n+1,k+1),dtype=int)
    S[:,0]=np.arange(n+1)*gap; S[0,:]=np.arange(k+1)*gap
    for i in range(1,n+1):
        for j in range(1,k+1):
            opts=(S[i-1,j-1]+(m if a[i-1]==b[j-1] else mm), S[i-1,j]+gap, S[i,j-1]+gap)
            P[i,j]=int(np.argmax(opts)); S[i,j]=max(opts)
    i,j=n,k; pairs=[]
    while i>0 and j>0:
        if P[i,j]==0: pairs.append((i-1,j-1)); i-=1; j-=1
        elif P[i,j]==1: i-=1
        else: j-=1
    return pairs[::-1]
rows=[]; summary={}
RUN=4  # require the match to sit inside a run of >=RUN consecutive identical-species aligned pairs
for s1,s2 in itertools.combinations(sorted(sess),2):
    a=sess[s1]; b=sess[s2]
    A=''.join(a.species.map(ab)); B=''.join(b.species.map(ab))
    pairs=nw(A,B)
    ok=[(i,j) for i,j in pairs if A[i]==B[j]]
    # runs of consecutive diagonal matches
    keep=set(); run=[]
    for p in pairs:
        if A[p[0]]==B[p[1]] and (not run or (p[0]==run[-1][0]+1 and p[1]==run[-1][1]+1)): run.append(p)
        else:
            if len(run)>=RUN: keep.update(run)
            run=[p] if A[p[0]]==B[p[1]] else []
    if len(run)>=RUN: keep.update(run)
    for i,j in sorted(keep):
        rows.append(dict(session_1=s1, file_1=a.orig[i], split_1=a.split_new[i], session_2=s2, file_2=b.orig[j], split_2=b.split_new[j], species=a.species[i]))
    summary[f"{s1} vs {s2}"]=dict(n1=len(A), n2=len(B), aligned_identical_species=len(ok), in_conserved_runs_ge4=len(keep))
m=pd.DataFrame(rows); m.to_csv(out+"/candidate_cross_session_matches.csv", index=False)
m['cross_split']=m.split_1!=m.split_2
test_files=set(c[c.split_new=='test'].orig)
tm = m[(m.file_1.isin(test_files)&(m.split_2!='test'))|(m.file_2.isin(test_files)&(m.split_1!='test'))]
tf = set(tm.file_1[tm.file_1.isin(test_files)]) | set(tm.file_2[tm.file_2.isin(test_files)])
summary['candidate_matches_total']=len(m); summary['candidate_matches_cross_split']=int(m.cross_split.sum())
summary['tierC_test_images_(non -2-)']=int((c.split_new=='test').sum())
summary['tierC_test_images_with_candidate_same_plant_in_train_or_val']=len(tf)
# null: shuffle species labels within session to estimate chance runs
rng=np.random.default_rng(0); null=[]
for r in range(20):
    tot=0
    for s1,s2 in itertools.combinations(sorted(sess),2):
        A=''.join(rng.permutation(list(''.join(sess[s1].species.map(ab))))); B=''.join(rng.permutation(list(''.join(sess[s2].species.map(ab)))))
        pairs=nw(A,B); run=[]; k=0
        for p in pairs:
            if A[p[0]]==B[p[1]] and (not run or (p[0]==run[-1][0]+1 and p[1]==run[-1][1]+1)): run.append(p)
            else:
                if len(run)>=RUN: k+=len(run)
                run=[p] if A[p[0]]==B[p[1]] else []
        if len(run)>=RUN: k+=len(run)
        tot+=k
    null.append(tot)
summary['null_shuffled_runs_ge4_mean']=float(np.mean(null)); summary['null_shuffled_runs_ge4_max']=int(np.max(null))
json.dump(summary, open(out+"/alignment_summary.json","w"), indent=2)
print(json.dumps(summary, indent=2))
