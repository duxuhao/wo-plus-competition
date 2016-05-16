# -*- coding: utf-8 -*-
"""
Created on Sat May 14 15:43:45 2016

@author: 21644336
"""

with sns.axes_style("dark"):
    for name in ['browseShoppingSites','browseITSites','browseFinanceSites','browseHealthSites','browseSportsSites']:
        #plt.hist(np.log(Pudong_Interest[name] + 1),alpha=0.7,log=True,histtype = 'stepfilled',label = name)
        sns.kdeplot(np.log10(Pudong_Interest[name] + 1), shade=True);

t = plt.legend(loc=1,fontsize= 15)
for text in t.get_texts():
    text.set_color('w')


plt.xlabel('Page View (10^)',fontsize = 14,color = 'w')
plt.ylabel('Normalize Quantity',fontsize = 14, color ='w')
plt.tick_params(axis="y", labelcolor="w")
plt.tick_params(axis="x", labelcolor="w")
plt.savefig('Interest2.png', dpi=1200,transparent=True)


Colname = [u'AutoApp',
       u'SocialNetworkSize', u'CrossProvinceExp', u'OverseaExp',
       u'browseShoppingSites', u'browseITSites', u'browseRestaurantSites',
       u'browseRealEstateSites', u'browseHealthSites', u'browseFinanceSites',
       u'browseTravelSites', u'browseSportsSites', u'browseAutoSites',
       u'browseNewsSites', u'browseCommunitySites', u'browseRecreationSites',
       u'browseJobsSites', u'browseEducationSites', 
       u'browseOnlineGamingSites']
per = []
for name in Colname:
    t = np.sum(~pd.isnull(Pudong_Interest[name])) / float(np.sum(~pd.isnull(df[name])) )
    per.append(np.sum(Pudong_Interest[name]) / t /np.nansum(df[name]))
    #per.append(np.nanmedian(Pudong_Interest[name].astype(float)) /np.nanmedian(df[name]))

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
t = sns.color_palette("husl", 8)
with sns.axes_style("dark"):
    plt.barh(np.arange(1,len(per)+1)-0.5,per,color = t)

plt.yticks(np.arange(1,len(per)+1), Colname)
plt.tick_params(axis="y", labelcolor="w")
plt.tick_params(axis="x", labelcolor="w")
plt.savefig('Interestdiff2.png', dpi=1200,transparent=True)


rs = np.random.RandomState(33)
d = Pudong_Interest[[u'AutoApp', u'FinanApp', u'StockApp',
       u'SocialNetworkSize', u'CrossProvinceExp', u'OverseaExp',
       u'browseShoppingSites', u'browseITSites', u'browseRestaurantSites',
       u'browseRealEstateSites', u'browseHealthSites', u'browseFinanceSites',
       u'browseTravelSites', u'browseSportsSites', u'browseAutoSites',
       u'browseNewsSites', u'browseCommunitySites', u'browseRecreationSites',
       u'browseJobsSites', u'browseEducationSites', 
       u'browseOnlineGamingSites']]
d.columns = range(21)
# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=2, yticklabels=2,
            linewidths=.8, cbar_kws={"shrink": .5}, ax=ax)
plt.tick_params(axis="y", labelcolor="w")
plt.tick_params(axis="x", labelcolor="w")
#sns.corrplot(d,  diag_names=False);

plt.savefig('Interestcorr2.png', dpi=1200,transparent=True)