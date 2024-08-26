function wdata = output(vbox,nx,ny,nz,nbox,name,varname)
fid = fopen(name, 'wb+');
ZONEMARKER= 299.0;
EOHMARKER = 357.0;
fwrite(fid, '#!TDV112');
fwrite(fid, 1, 'int');
fwrite(fid, 0, 'int');
fwrite(fid, 0, 'int')
fwrite(fid, nbox, 'int');
for ii = 1 : nbox
    dumstring(fid,varname(ii));
end
fwrite(fid, ZONEMARKER, 'single');
dumstring(fid,"ZONE 001")
fwrite(fid, -1, 'int');
fwrite(fid, -1, 'int');
fwrite(fid, 0, 'int');
fwrite(fid, 0, 'int');
fwrite(fid, -1, 'int');
fwrite(fid, 0, 'int');
fwrite(fid, 0, 'int');
fwrite(fid, 0, 'int');
fwrite(fid, 0, 'int');
fwrite(fid, nx, 'int');
fwrite(fid, ny, 'int');
fwrite(fid, nz, 'int');
fwrite(fid, 0, 'int');
fwrite(fid, EOHMARKER, 'single');
fwrite(fid, ZONEMARKER, 'single');
for ii = 1 : nbox
    fwrite(fid, 1, 'int');
end
fwrite(fid, 0, 'int');
fwrite(fid, 0, 'int');
fwrite(fid, -1, 'int');
for ii=1 : nbox
    fwrite(fid, min(min(min(vbox(:,:,:,ii)))), 'double');
    fwrite(fid, max(max(max(vbox(:,:,:,ii)))), 'double');
end
fwrite(fid, vbox, 'single');
fclose(fid);
wdata = 'write data successfully';
end

function dumstring(fid, strings)
chars = char(strings);
[~,lens] = size(chars);
for ii = 1:lens
    fwrite(fid, chars(ii), 'int');
end
fwrite(fid, 0, 'int');
end