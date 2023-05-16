{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "import csv\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.cluster import MeanShift, estimate_bandwidth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Load data from input file\n",
    "input_file = 'sales_data_sample.csv'\n",
    "file_reader = csv.reader(open(input_file, 'r'), delimiter =',')\n",
    "\n",
    "X=[]\n",
    "for count, row in enumerate(file_reader):\n",
    "    if not count:\n",
    "        names = row[1:]\n",
    "        continue\n",
    "\n",
    "    X.append([float(x)for x in row [1:]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Found array with 0 feature(s) (shape=(2822, 0)) while a minimum of 1 is required.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[25], line 5\u001b[0m\n\u001b[0;32m      2\u001b[0m X\u001b[39m=\u001b[39mnp\u001b[39m.\u001b[39marray(X)\n\u001b[0;32m      4\u001b[0m \u001b[39m#Estimating the bandwidth of input data\u001b[39;00m\n\u001b[1;32m----> 5\u001b[0m bandwidth\u001b[39m=\u001b[39m estimate_bandwidth(X, quantile\u001b[39m=\u001b[39;49m\u001b[39m0.8\u001b[39;49m, n_samples\u001b[39m=\u001b[39;49m\u001b[39mlen\u001b[39;49m(X))\n\u001b[0;32m      7\u001b[0m \u001b[39m#Compute clustering with MeanShift\u001b[39;00m\n\u001b[0;32m      8\u001b[0m meanshift_model\u001b[39m=\u001b[39m MeanShift(bandwidth\u001b[39m=\u001b[39mbandwidth, bin_seeding\u001b[39m=\u001b[39m\u001b[39mTrue\u001b[39;00m)\n",
      "File \u001b[1;32mc:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\sklearn\\cluster\\_mean_shift.py:67\u001b[0m, in \u001b[0;36mestimate_bandwidth\u001b[1;34m(X, quantile, n_samples, random_state, n_jobs)\u001b[0m\n\u001b[0;32m     32\u001b[0m \u001b[39mdef\u001b[39;00m \u001b[39mestimate_bandwidth\u001b[39m(X, \u001b[39m*\u001b[39m, quantile\u001b[39m=\u001b[39m\u001b[39m0.3\u001b[39m, n_samples\u001b[39m=\u001b[39m\u001b[39mNone\u001b[39;00m, random_state\u001b[39m=\u001b[39m\u001b[39m0\u001b[39m, n_jobs\u001b[39m=\u001b[39m\u001b[39mNone\u001b[39;00m):\n\u001b[0;32m     33\u001b[0m \u001b[39m    \u001b[39m\u001b[39m\"\"\"Estimate the bandwidth to use with the mean-shift algorithm.\u001b[39;00m\n\u001b[0;32m     34\u001b[0m \n\u001b[0;32m     35\u001b[0m \u001b[39m    That this function takes time at least quadratic in n_samples. For large\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     65\u001b[0m \u001b[39m        The bandwidth parameter.\u001b[39;00m\n\u001b[0;32m     66\u001b[0m \u001b[39m    \"\"\"\u001b[39;00m\n\u001b[1;32m---> 67\u001b[0m     X \u001b[39m=\u001b[39m check_array(X)\n\u001b[0;32m     69\u001b[0m     random_state \u001b[39m=\u001b[39m check_random_state(random_state)\n\u001b[0;32m     70\u001b[0m     \u001b[39mif\u001b[39;00m n_samples \u001b[39mis\u001b[39;00m \u001b[39mnot\u001b[39;00m \u001b[39mNone\u001b[39;00m:\n",
      "File \u001b[1;32mc:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\sklearn\\utils\\validation.py:940\u001b[0m, in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)\u001b[0m\n\u001b[0;32m    938\u001b[0m     n_features \u001b[39m=\u001b[39m array\u001b[39m.\u001b[39mshape[\u001b[39m1\u001b[39m]\n\u001b[0;32m    939\u001b[0m     \u001b[39mif\u001b[39;00m n_features \u001b[39m<\u001b[39m ensure_min_features:\n\u001b[1;32m--> 940\u001b[0m         \u001b[39mraise\u001b[39;00m \u001b[39mValueError\u001b[39;00m(\n\u001b[0;32m    941\u001b[0m             \u001b[39m\"\u001b[39m\u001b[39mFound array with \u001b[39m\u001b[39m%d\u001b[39;00m\u001b[39m feature(s) (shape=\u001b[39m\u001b[39m%s\u001b[39;00m\u001b[39m) while\u001b[39m\u001b[39m\"\u001b[39m\n\u001b[0;32m    942\u001b[0m             \u001b[39m\"\u001b[39m\u001b[39m a minimum of \u001b[39m\u001b[39m%d\u001b[39;00m\u001b[39m is required\u001b[39m\u001b[39m%s\u001b[39;00m\u001b[39m.\u001b[39m\u001b[39m\"\u001b[39m\n\u001b[0;32m    943\u001b[0m             \u001b[39m%\u001b[39m (n_features, array\u001b[39m.\u001b[39mshape, ensure_min_features, context)\n\u001b[0;32m    944\u001b[0m         )\n\u001b[0;32m    946\u001b[0m \u001b[39mif\u001b[39;00m copy:\n\u001b[0;32m    947\u001b[0m     \u001b[39mif\u001b[39;00m xp\u001b[39m.\u001b[39m\u001b[39m__name__\u001b[39m \u001b[39min\u001b[39;00m {\u001b[39m\"\u001b[39m\u001b[39mnumpy\u001b[39m\u001b[39m\"\u001b[39m, \u001b[39m\"\u001b[39m\u001b[39mnumpy.array_api\u001b[39m\u001b[39m\"\u001b[39m}:\n\u001b[0;32m    948\u001b[0m         \u001b[39m# only make a copy if `array` and `array_orig` may share memory`\u001b[39;00m\n",
      "\u001b[1;31mValueError\u001b[0m: Found array with 0 feature(s) (shape=(2822, 0)) while a minimum of 1 is required."
     ]
    }
   ],
   "source": [
    "#Convert to numpy  array\n",
    "X=np.array(X)\n",
    "\n",
    "#Estimating the bandwidth of input data\n",
    "bandwidth= estimate_bandwidth(X, quantile=0.8, n_samples=len(X))\n",
    "\n",
    "#Compute clustering with MeanShift\n",
    "meanshift_model= MeanShift(bandwidth=bandwidth, bin_seeding=True)\n",
    "meanshift_model.fit(X)\n",
    "labels= meanshift_model.labels_\n",
    "cluster_centers= meanshift_model.cluster_centers_\n",
    "num_cluster= len(np.unique(labels))\n",
    "\n",
    "print(\"\\nNumber of cluster in input data = \", num_clusters)\n",
    "\n",
    "print(\"\\nCenters of clusters:\")\n",
    "print('\\t'.joint([name[:3]for name in names]))\n",
    "for cluster_centers in cluster_centers:\n",
    "    print('\\t'.joint([str[int(x)]for x in cluster_centers]))\n",
    "\n",
    "#Extract  two features for visualization\n",
    "cluster_centers_2d= cluster_centers[:, 1:3]\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Plot the cluster centers\n",
    "plt.figure()\n",
    "plt.scatter(cluster_centers_2d[:,0], cluster_centers_2d[:,1],s=120, edgecolors='black', facecolors='none')\n",
    "\n",
    "offset=0.25\n",
    "plt.xlim(cluster_centers_2d[:,0].min()-offset *cluster_centers_2d[:,0].ptp(), cluster_centers_2d[:,0].max()+offset *cluster_centers_2d[:0].ptp(),)\n",
    "plt.ylim(cluster_centers_2d[:,1].min()-offset *cluster_centers_2d[:,1].ptp(), cluster_centers_2d[:,1].max()+offset *cluster_centers_2d[:,1].ptp(),)\n",
    "\n",
    "plt.title('Centers of 2D clusters')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
