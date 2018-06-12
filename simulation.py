#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import functions as fu
from scipy import special
from scipy import stats
import traceback
import sys
	
	
def run_sims_window(win,sd,periods,d,nsims,adj,mixed_norm,cluster,minsd,p,sd_arr,sds):
	a=[]
	sd_arr=sd_arr*sd
	pmsq=np.sum(p*sd_arr**2)**0.5

	smean=np.mean(sds)
	#avg_tz_bias,void=calc_adj(win, cluster, pmsq*(smean/sd),d,p,sd_arr*(smean/sd))
	
	#for i in [1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960, 1920, 3600, 7200, 14400, 28800]:
	#	print(cluster_inflation(p,i,cluster,periods,sd_arr))
		
	(ineff_c_abs, ineff_c_sq) =cluster_inflation(p,win,cluster,periods,sd_arr,d)

	tz_bias,E_abs=calc_adj(win, cluster, pmsq, d, p, sd_arr)
	tz_bias2,E_abs2=calc_adj(win-1, cluster, pmsq, d, p, sd_arr)
	for i in range(nsims):
		r=simulation(1E-10, sd, periods, d, win,adj,mixed_norm,cluster,p,sd_arr,E_abs,E_abs2,pmsq,1,tz_bias)
		a.append(r)	
	a=np.array(a)
	
	a=a+(a==0)*minsd
	a_ln=np.log(a)
	m=np.mean(a_ln,0)
	
	ssd=(np.sum((a_ln-m)**2,0)/(nsims-1))**0.5
	ssd=ssd
	#print((np.mean(a,0),E_abs))
	m_arr=np.concatenate((m,ssd,np.abs(m),[ineff_c_sq,ineff_c_abs]))
	m_arr=m_arr.reshape((len(m_arr),1))
	print (ssd)
	print((ineff_c_abs, ineff_c_sq))
	return m_arr




def calc_adj(win,cluster,pmsq,d,p,sd_arr):
	if win==0:
		return 1,np.sum(p*sd_arr)
	if cluster>1:
		if win>cluster:
			
			wc=win/cluster
			var1,P1=prob_space(p, sd_arr, int(wc)+1)
			var2,P2=prob_space(p, sd_arr, int(wc)+2)
			tz1=ticksizebias(var1*(win/(int(wc)+1)), P1, win, pmsq, d)
			tz2=ticksizebias(var2*(win/(int(wc)+2)), P2, win, pmsq, d)
			E1=E_abs_calc(var1,P1,int(wc)+1)
			E2=E_abs_calc(var2,P2,int(wc)+2)
			x=(wc-int(wc))
			E = (1-x)*E1+x*E2
			tz = (1-x)*tz1+x*tz2
		else:
			var,P=prob_space(p, sd_arr, 1)
			tz=ticksizebias(var*win, P, win, pmsq, d)
			E=E_abs_calc(sd_arr**2,p,1)
	else:
		var,P=prob_space(p, sd_arr, win)
		tz=ticksizebias(var, P, win, pmsq, d)
		E = E_abs_calc(var,P,win)
	return tz,E

def cluster_inflation(p,win,cluster,T,sd_arr,d):
	#if cluster==1:
	#	return 1

	var=np.sum(p*sd_arr**2)
	sd=np.sum(p*sd_arr)
	
	N=int(T/win)
	Es1,Es2=EsN(p, sd_arr, win,1,var)
	
	ineff_c_sq1=((np.pi/4)**0.5*d*Es1-Es2*(4/np.pi)**0.5)/Es1**2
	ineff_c_sq2=max(((cluster/win),1))*((Es2/Es1**2)-1)
	ineff_c_sq=(ineff_c_sq1+ineff_c_sq2)/N
	ineff_c_sq05=ineff_c_sq**0.5
	
	tzbias3=d*((np.pi/8)**0.5)/(sd*T**0.5)
	
	Es4,E_abs_sq_inv=Esd4(p, sd_arr, win,1,var)
	
	
	Es4_cluster,E_abs_sq_inv_cluster=Esd4(p, sd_arr,win,cluster,var)

	ineff_c_sq=(2*Es4_cluster+max(((cluster/win),1))*((Es4_cluster)-1))/N
	ineff_sq=(3*Es4-1)/N
	ineff_r=(ineff_c_sq/ineff_sq)**0.5
	
	var_arr,P=prob_space(p, sd_arr, int(cluster/win))
	tzbias4=np.sum(P*d*((np.pi/8)**0.5)/((var_arr*win/cluster)**0.5*T**0.5))
	tzbias2=d*((np.pi/8)**0.5)/(var**0.5*T**0.5)
	tzbias3=d*((np.pi/8)**0.5)/(sd*T**0.5)

	
	
	ineff_c_abs=(max(((cluster/win),1))*(E_abs_sq_inv_cluster-1)+E_abs_sq_inv_cluster*(np.pi*0.5-1))/N
	ineff_abs=((E_abs_sq_inv-1)+E_abs_sq_inv*(np.pi*0.5-1))/N
	ineff_abs_r=(ineff_c_abs/ineff_abs)**0.5
	#print(((E_abs_sq_inv_cluster/E_abs_sq_inv)**2,win))
	

	ineff_c_abs2=max((ineff_c_abs**0.5,tzbias2))
	ineff_c_sq2=max((ineff_c_sq**0.5,tzbias2))
	return (ineff_c_abs2,ineff_c_sq2)
	
def Esd4(p, sd_arr, win,cluster,v):
	eff_win=max((win/cluster,1))
	eff_win=int(np.round(eff_win,0))
	var,P=prob_space(p, sd_arr, eff_win)
	var=var/(eff_win)
	r=np.sum(P*var**2)/v**2
	
	E_abs_sq_inv=v/np.sum(P*var**0.5)**2
	
	return r, E_abs_sq_inv
	
	
def EsN(p, sd_arr, win,cluster,v):
	eff_win=max((win/cluster,1))
	eff_win=int(np.round(eff_win,0))
	var,P=prob_space(p, sd_arr, eff_win)
	Es1=np.sum(P*var**0.5)
	Es2=np.sum(P*var)

	
	return Es1, Es2


def prob_space(p,sd_arr,win):
	"""returns the set of outcomes (variances) var and its probabilities P"""
	sd_arrsq=sd_arr**2
	if type(sd_arr)==float or type(sd_arr)==np.float64 or win>450:
		var=np.sum(p*sd_arr**2)*win
		return var, 1
	K=len(sd_arr)
	a=np.rollaxis(np.indices([win+1]*(K-1)),0,K).reshape((win+1)**(K-1),K-1)
	a=a[np.sum(a,1)<=win]
	b=win-np.sum(a,1).reshape((len(a),1))
	Q=np.concatenate((a,b),1)
	P=multinominaldist(Q,p)
	var=np.sum(Q*sd_arrsq,1)
	return var,P

def E_abs_calc(var,P,win):
	E_sd=np.sum(P*var**0.5)/win**0.5
	
	return E_sd

def ticksizebias(var,P,win,pmsq,d):
	tz=asympt_var(d, var)
	tz=np.sum(P*tz)**0.5/(pmsq*win**0.5)
	return tz
	

def multinominaldist(q,p):
	a=special.gammaln(np.sum(q[0])+1)
	b=np.sum(special.gammaln(q+1),1)
	c=np.prod(p**q,1)

	return np.exp(a-b)*c



def random_mixed_normal(mu,N,sd_arr,p,cluster):
	"""sd_arr should be an array of standard deviations such that np.sum(p*np.array(sd_arr))=1"""
	if cluster>1:
		N_clusters=int(N/cluster)
		r=np.random.uniform(size=N_clusters).reshape((N_clusters,1))
		r=r*np.ones((1,cluster))
	else:
		r=np.random.uniform(size=N)
	r=r.reshape((N,1))
	p1=np.cumsum(p).reshape((1,len(p)))
	p1[0,2]=1.1
	p0=np.append([[-0.1]],p1[0,:-1])
	n=np.nonzero((r>p0)*(r<=p1))
	sd_rm=sd_arr[n[1]]
	rnd=np.random.normal(mu,sd_rm,N)

	return rnd
	
def return_calc(p,d,N,windows,win,p0):	
	if d==0:
		p_obs=p
	else:
		p0=np.round(p0/d)*d
		rnded=np.round(p/d)
		p_obs=rnded*d
	lp_obs=np.log(p_obs)

	lp_obs=lp_obs.reshape((windows,win)).T
	ret_period=lp_obs[-1]-np.roll(lp_obs[-1],1)
	ret_period[0]=lp_obs[-1,0]-np.log(p0)

	return lp_obs,ret_period



	
def sim_returns(mixed_norm,mu,sd,N,d,win,cluster,p,sd_arr):
	windows=int(N/win)
	N=windows*win	
		
	if mixed_norm:
		rnd=random_mixed_normal(mu,N,sd_arr,p,cluster)
	else:
		rnd=np.random.normal(mu,sd,N)
	
	#if rho>0: not in use, clustering used instead
	#	rnd=rnd-rho*np.roll(rnd,1)
	
	p0=1+np.random.uniform(-d, d)#ensures it does not allways start far from a changing value
	p_cont=np.exp(np.cumsum(rnd))
	p_cont=p_cont*p0
	
	lp_obs,ret_period=return_calc(p_cont,d,N,windows,win,p0)
	psd=np.sum(p*sd_arr)

	return lp_obs,ret_period,sd_arr,p,windows,psd
	
	
	
def simulation(mu,sd,N,d,win,adj,mixed_norm,cluster,p,sd_arr,E_abs,E_abs2,pmsq, avg_tz_bias,tz_bias):
	
	lp_obs,ret_period,sd_arr,p,windows,psd=sim_returns(mixed_norm,mu,sd,N,d, win,cluster,p,sd_arr)
	

	rng=rng_func(lp_obs,win,adj,mixed_norm,psd,cluster,p,sd_arr,E_abs2,sd)

	msq_adj,msq_unadj,avg_abs,msq1,msq2=msq_func(win,windows,ret_period,adj,pmsq,sd_arr,d,p,psd,E_abs, avg_tz_bias,tz_bias,sd)
	
	return np.array([rng,msq_adj,msq_unadj,avg_abs,msq1,msq2])


def asympt_var(d,var):
	
	sd=var**0.5
	if d==0:
		v=1
	v1=var+(d**2)/6.0
	v2=2*(d*sd)/((2.0*np.pi)**0.5)
	cond=d/sd<2.5
	v=cond*v1+(cond==0)*v2
	return v


def msq_func(win,windows,ret,adj,pmsq,sd_arr,d,p,psd,E_abs, avg_tz_bias,tz_bias,sd):
	if windows==1:
		return np.nan,np.nan,np.nan
	freq=len(np.nonzero(ret==0)[0])/len(ret)
	ret=ret-np.sum(ret)/(windows)
	ln_bias=(2*np.exp(special.psi((windows-1)/2))/(windows-1))**0.5

	dofa=dof_adj(windows-1)
	
	
	msq=(np.sum(ret**2)/(win*(windows-1)))
	
	avg_abs=(np.sum(np.abs(ret))/win**0.5)/(((windows-1)*(windows))**0.5)

	abs_bias=ln_bias*(2/np.pi)**0.5/dofa
	
	avg_abs=avg_abs/abs_bias
	if adj:
		avg_abs=avg_abs/E_abs
	else:
		avg_abs=avg_abs/sd
	
	msq_adj=msq/(ln_bias*tz_bias*pmsq)**2
	msq_unadj=msq/(ln_bias*pmsq)**2
	msq_adj2=msq/(tz_bias*pmsq)**2
	return msq_adj,msq_unadj,avg_abs,msq,msq**2



def dof_adj(k):

	if k>250:
		a=1
	else:
		if k<1:
			k=1
		a=2**0.5*special.gamma((k+1)/2)/(special.gamma(k/2))	
		a=a/k**0.5
	return a

	
def rng_func(data,win, adj,mixed_norm,psd,cluster,p,sd_arr,E_abs,sd):
	if win==1:
		return np.nan	

	mx=np.max(data,0)
	mn=np.min(data,0)
	as_exp=(8/np.pi)**0.5
	rng=np.mean(mx-mn)/(as_exp*(win-1)**0.5)
	if not adj:
		return rng/sd
	a=np.exp(-(win-2)**0.5*0.267)
	ret=rng/(E_abs*(2-a))
	return ret



