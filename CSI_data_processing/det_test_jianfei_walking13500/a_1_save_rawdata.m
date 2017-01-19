clear all;
close all;
No_datapacket=600;
package_num =2;
packet_count=1;
time_window=3;
no_packet_effect=1;
packet_actual=1;
for packet_count=1:No_datapacket
    fileName = strcat(['8866-', int2str(packet_count)]'')
    
    read_log_file_be (fileName)
    Data = ans;
    %     packet analysis
    packet1 = Data{1};
    if isempty(packet1)==0
        if isempty(packet1.csi)==0 %&& no_packet_effect>=time_window+1
            if mean(squeeze(packet1.csi(1,1,:))')~=0
                csidata = packet1.csi;
                csirawdata_all{no_packet_effect} = packet1.csi;
                tx=1;
                total_rx=3;
                total_subcarrier=length(csirawdata_all{no_packet_effect});
                % get useful raw data from rx
                %                 no_packet_effect=1;
                for no_rx=1:total_rx
                    for no_packet=1:length(csirawdata_all)
                        csi_temp{no_rx}(no_packet_effect,:)=squeeze(csirawdata_all{no_packet}(no_rx,tx,:))';
                    end
                end
                no_packet_effect=no_packet_effect+1;
            else
            end
        else
        end
    else
    end
end
save csirawdata_all csirawdata_all
% calculate raw amp and phase
for  no_packet=1:packet_count
    for select_subcarrier=1:total_subcarrier
    for no_rx=1:total_rx
csiraw_amp{no_rx}(no_packet,select_subcarrier)=db(squeeze(csirawdata_all{no_packet}(no_rx,tx,select_subcarrier))');
csiraw_phase{no_rx}(no_packet,select_subcarrier)=unwrap(angle(squeeze(csirawdata_all{no_packet}(no_rx,tx,select_subcarrier))'));
    end
    end
end
save csiraw_amp csiraw_amp
save csiraw_phase csiraw_phase

% % plot rawdata
% for  no_packet=1:packet_count
%     for no_rx=1:total_rx
%         subplot(3,2,no_rx*2-1)
%         plot3(zeros(1,total_subcarrier)+no_packet,1:total_subcarrier,db(squeeze(csirawdata_all{no_packet}(no_rx,tx,:))'));
%         xlabel('Time');
%         ylabel('Subcarrier');
%         zlabel('[dB]');
%         hold on;
%         subplot(3,2,no_rx*2)
%         plot3(zeros(1,total_subcarrier)+no_packet,1:total_subcarrier,unwrap(angle(squeeze(csirawdata_all{no_packet}(no_rx,tx,:))')));
%         xlabel('Time');
%         ylabel('Subcarrier');
%         zlabel('[Rad]');
%         hold on;
%     end
% end
% 
% savefig('a_rawdata.fig')
% % plot rawdata of one select_subcarrier
% 

