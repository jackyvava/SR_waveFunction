# 增加迭代步数

## 迭代步数为10001

运行时间: 1378.4957 秒
相对误差: 0.031582

![velocity_fields_step10000_0.0000](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\velocity_fields_step10000_0.0000.png )

![wave_functions_step10000_0.0000](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\wave_functions_step10000_0.0000.png)

# 增加噪音

这一部分的迭代步数为5001

## 无噪音

基础的 TG 涡流速度场，速度场是光滑且确定的，没有随机性

```matlab
function [vx, vy] = TGVelocityOneForm(obj)
	%%% initial TG flow in 2D
	vx = obj.dx * sin(obj.px + 0.5 * obj.dx) .* cos(obj.py);
	vy = -obj.dx * cos(obj.px) .* sin(obj.py + 0.5 * obj.dy);
end
```

![velocity_fields](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\velocity_fields_0.0000.png)

![wave_functions](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\wave_functions_0.0000.png)

运行时间: 550.5405 秒
相对误差: 0.097146

## 增加标准差为0.0001的噪音

通过指定`noise`来设置噪音标准差

```matlab
function [vx, vy] = TGVelocityOneForm_noise(obj, noise)
%%% initial TG flow in 2D with random noise
	noise_level = 0.0001; % 设置噪声水平，可以根据需要调整
	vx = obj.dx * sin(obj.px + 0.5 * obj.dx) .* cos(obj.py) + noise_level * randn(size(obj.px));
	vy = -obj.dx * cos(obj.px) .* sin(obj.py + 0.5 * obj.dy) + noise_level * randn(size(obj.py));
end
```

运行时间: 
相对误差: 0.041854

![velocity_fields_0.0001](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\velocity_fields_0.0001.png)

![wave_functions_0.0001](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\wave_functions_0.0001.png)

## 增加标准差为0.001的噪音

运行时间: 884.2416 秒
相对误差: 0.040767

![velocity_fields_0.0001](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\velocity_fields_0.0010.png)

![wave_functions_0.0001](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\wave_functions_0.0010.png)

## 增加标准差为0.01的噪音

运行时间: 856.0888 秒
相对误差: 0.43336

![velocity_fields_0.0001](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\velocity_fields_0.0100.png)

![wave_functions_0.0001](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\wave_functions_0.0100.png)

## 增加标准差为0.05的噪音

运行时间: 918.6934 秒
相对误差: 0.5763

![velocity_fields_step10000_0.0500](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\velocity_fields_step10000_0.0500.png)

![wave_functions_step10000_0.0500](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\wave_functions_step10000_0.0500.png)

## 增加标准差为0.1的噪音

运行时间: 944.669 秒
相对误差: 0.6329

![velocity_fields_0.0001](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\velocity_fields_0.1000.png)

![wave_functions_0.0001](D:\zjPhD\Programzj\psiToU\Ginzburg_Lan\code5\wave_functions_0.1000.png)

# 总结

|          | no noise | 0.0001   | 0.001    | 0.01     | 0.05     | 0.1     |
| -------- | -------- | -------- | -------- | -------- | -------- | ------- |
| Time(s)  | 550.5405 | 807.3557 | 884.2416 | 856.0888 | 918.6934 | 944.669 |
| 相对误差 | 0.097146 | 0.040767 | 0.040767 | 0.43336  | 0.5763   | 0.6329  |

通过上述的研究，可以看到随着标准差的增加，0.05以下波函数还是基本能够恢复，但是再增加到0.1，波函数也趋向于噪声分布。