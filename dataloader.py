import cdflib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np

cdf  = cdflib.CDF('D:/Dataset______________/tec/tec_2011/gps_tec1hr_igs_20110101_v01.cdf')

tec  = cdf.varget('tecUHR')   #.varget提取出指定变量 或者 tecEHR / tecCOD / tecCOR

lat  = cdf.varget('lat')
print(lat)
print('lat shape :', lat.shape)  #lat shape : (71,)

lon = cdf.varget('lon')         #lon shape : (73,)
img = Image.fromarray(tec[0,:,:])

print(tec[0,:,:])
img.show()
print('tec[0,:,:] shape :',tec[0,:,:].shape)
print('tec[0,:,:] 类型 :',type(tec[0,:,:]))
#exit()

print('tec shape :', tec.shape)   # 预期 (time, lat, lon)
print("tec size :", tec.size)
print('lat shape :', lat.shape)

print('lon shape :', lon.shape)
print(type(lon))                  # 查看类型<class 'numpy.ndarray'>
print(lon.dtype)                   #查看数据类型  float32
exit()
plt.show()

plt.figure(figsize=(10,5))
#proj = ccrs.PlateCarree()
plt.pcolormesh(lon, lat, tec[0, :, :], shading='auto', cmap='plasma')

plt.colorbar(label='TECU')
plt.title(f'Global TEC (tecUHR) – {0}st epoch')
plt.savefig(f'tecUHR_{0}.png', dpi=150)
plt.show()

proj = ccrs.PlateCarree()          # 等经纬度投影
fig  = plt.figure(figsize=(9, 4.5))
ax   = plt.axes(projection=proj)

mesh = ax.pcolormesh(lon, lat, tec[1, :, :], transform=proj,
                     shading='auto', cmap='jet')
ax.coastlines()                    # 海岸线
ax.gridlines(draw_labels=True)
ax.set_global()                    # 显示全球
fig.colorbar(mesh, ax=ax, label='TECU', orientation='horizontal')
plt.show()