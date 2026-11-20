"""PickleTrack AI: upload, track players, label rally outcomes and export statistics."""
import io
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from analysis import video_info, preview_frame, track_people, total_distance, COURT_W, COURT_L

st.set_page_config(page_title='PickleTrack AI', page_icon='🏓', layout='wide')
st.markdown('''<style>
.block-container{padding-top:1.6rem;max-width:1450px}
.hero{padding:22px 25px;background:linear-gradient(110deg,#1d323d,#153b39);border:1px solid #445e61;border-radius:17px;margin-bottom:20px}
.hero h1{margin:0;color:#c8f576;font-size:2.2rem}.hero p{margin:6px 0 0;color:#d9e4e8}
.small{font-size:0.85rem;color:#adbfc6}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🏓 PickleTrack AI</h1><p>Video-to-court movement analytics · Confirmed missed-return zones · Player performance</p></div>',unsafe_allow_html=True)

if 'events' not in st.session_state: st.session_state.events=[]
if 'tracking' not in st.session_state: st.session_state.tracking=None
if 'video_id' not in st.session_state: st.session_state.video_id=None

with st.sidebar:
    st.header('Match settings')
    match_name=st.text_input('Match name','Pickleball practice')
    players=st.text_area('Player names (one per line)','Player A\nPlayer B').splitlines()
    players=[p.strip() for p in players if p.strip()]
    if not players: players=['Player A']
    st.info('Best results: fixed high/wide camera showing the complete court. Upload consented match footage.')

source=st.radio('Video source', ['Upload video', 'YouTube link'], horizontal=True)
file=None
youtube_video=None
if source == 'Upload video':
    file=st.file_uploader('Upload match video (MP4 / MOV / AVI)', type=['mp4','mov','avi'])
else:
    from urllib.parse import urlparse, parse_qs
    url=st.text_input('YouTube video URL', placeholder='https://www.youtube.com/watch?v=...').strip()
    st.caption('Public videos only. Use footage you are permitted to analyse. Cloud downloads can be blocked by YouTube.')
    if st.button('Load YouTube video'):
        parts=urlparse(url)
        host=(parts.hostname or '').lower()
        allowed={'youtube.com','www.youtube.com','m.youtube.com','youtu.be','www.youtu.be','www.youtube-nocookie.com'}
        valid=parts.scheme=='https' and host in allowed
        if not valid:
            st.error('Please enter a valid HTTPS YouTube video URL.')
        else:
            try:
                import yt_dlp
                import hashlib
                key=hashlib.sha256(url.encode('utf-8')).hexdigest()[:18]
                target_base=Path(tempfile.gettempdir())/f'pickletrack_youtube_{key}'
                existing=list(Path(tempfile.gettempdir()).glob(f'pickletrack_youtube_{key}.*'))
                target=existing[0] if existing else None
                if target is None or not target.exists():
                    with st.spinner('Retrieving an available video format from YouTube...'):
                        opts={
                            'format':'best[height<=720]/best',
                            'outtmpl':str(target_base)+'.%(ext)s',
                            'noplaylist':True,
                            'quiet':True,
                            'no_warnings':True,
                            'max_filesize':250*1024*1024,
                            'match_filter':yt_dlp.utils.match_filter_func('duration <= 900'),
                            'socket_timeout':20,
                            'retries':2,
                        }
                        with yt_dlp.YoutubeDL(opts) as downloader:
                            downloader.download([url])
                        downloaded=list(Path(tempfile.gettempdir()).glob(f'pickletrack_youtube_{key}.*'))
                        target=downloaded[0] if downloaded else None
                if target is None or not target.exists() or not target.stat().st_size or target.stat().st_size>250*1024*1024:
                    if target is not None: target.unlink(missing_ok=True)
                    raise ValueError('Video unavailable, over 15 minutes, or larger than 250 MB.')
                st.session_state.youtube_video={'path':str(target),'url':url}
                st.success('YouTube video loaded. You can now calibrate the court and analyse it.')
            except Exception as exc:
                st.session_state.youtube_video=None
                st.error(f'YouTube video could not be loaded: {exc}')
                st.info('YouTube can restrict access from cloud servers, even when the URL works in a browser. If it continues failing, use an authorised video file with Upload video.')
    youtube_video=st.session_state.get('youtube_video')
    if youtube_video and url!=youtube_video['url']:
        youtube_video=None

if file is not None or youtube_video:
    import hashlib
    if file is not None:
        b=file.getvalue()
        signature=hashlib.sha256(b).hexdigest()
        suffix=Path(file.name).suffix.lower()
        path=Path(tempfile.gettempdir())/f'pickletrack_{signature[:18]}{suffix}'
        if not path.exists() or path.stat().st_size!=len(b): path.write_bytes(b)
        player_video=b
    else:
        path=Path(youtube_video['path'])
        signature='youtube:'+youtube_video['url']
        player_video=str(path) if path.suffix.lower()=='.mp4' else None
        if not path.exists():
            st.error('Temporary YouTube video expired. Click Load YouTube video again.')
            st.stop()
    if signature != st.session_state.video_id:
        st.session_state.video_id=signature
        st.session_state.events=[]
        st.session_state.tracking=None
    try:
        info=video_info(path); frame=preview_frame(path)
    except Exception as exc:
        st.error(str(exc)); st.stop()
    col1,col2=st.columns([1.25,1])
    with col1:
        st.subheader('Match footage')
        if player_video is not None: st.video(player_video)
        else: st.info('Video retrieved for analysis. Playback preview may not support this format; court calibration still works.')
        st.caption(f'{info["width"]}×{info["height"]} • {info["seconds"]:.1f}s • {info["fps"]:.1f} fps')
    with col2:
        st.subheader('Court calibration')
        st.caption('Enter pixel coordinates for all 4 outside court corners. Order: NEAR left → NEAR right → FAR right → FAR left. Preview below.')
        h,w=frame.shape[:2]
        defaults=[(int(w*.16),int(h*.90)),(int(w*.84),int(h*.90)),(int(w*.60),int(h*.20)),(int(w*.40),int(h*.20))]
        labels=['Near left','Near right','Far right','Far left']
        corners=[]
        for label,(dx,dy) in zip(labels,defaults):
            c1,c2=st.columns(2)
            with c1: x=st.number_input(label+' X',0,w-1,dx,key=label+'x')
            with c2: y=st.number_input(label+' Y',0,h-1,dy,key=label+'y')
            corners.append((x,y))
        import cv2
        overlay=frame.copy()
        pts=np.array(corners,np.int32)
        cv2.polylines(overlay,[pts],True,(194,255,60),3)
        for i,pt in enumerate(corners):
            cv2.circle(overlay,pt,6,(255,200,0),-1)
            cv2.putText(overlay,str(i+1),(pt[0]+8,pt[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        st.image(overlay,caption='Adjust corner coordinates until green perimeter matches the real sidelines and baselines.',use_container_width=True)
    st.divider()
    st.subheader('1 · AI movement analysis')
    s1,s2,s3=st.columns(3)
    with s1: sample_rate=st.select_slider('Analyse every N frames',options=[1,2,3,5,8],value=3)
    with s2: max_seconds=st.slider('Max seconds to process',10,max(10,int(info['seconds'])),min(120,max(10,int(info['seconds']))))
    with s3: threshold=st.slider('Person confidence',.15,.8,.3,.05)
    if st.button('▶ Analyse player movements',type='primary'):
        bar=st.progress(0)
        try:
            data=track_people(path,corners,every_n=sample_rate,conf=threshold,max_seconds=max_seconds,progress=bar.progress)
            st.session_state.tracking=data
            st.success(f'Completed: {len(data):,} tracked court positions. Check IDs below.')
        except Exception as exc: st.error(f'Could not run AI analysis: {exc}')
    tracks=st.session_state.tracking
    if tracks is not None:
        if tracks.empty:
            st.warning('No court-side players detected. Check the corners, video, and confidence setting.')
        else:
            counts=tracks.track_id.value_counts()
            st.caption('Tracking IDs are automatic and may change when players overlap or leave frame. Verify the ID visually before assigning it to a player.')
            ids=counts.index.tolist()
            chosen=st.selectbox('Choose track ID for movement heatmap',ids,format_func=lambda x:f'ID {x} · {counts[x]} observations')
            selected=tracks[tracks.track_id==chosen].copy()
            c1,c2,c3=st.columns(3)
            c1.metric('Tracked observations',len(selected))
            c2.metric('Estimated distance',f'{total_distance(selected):.1f} m')
            c3.metric('Tracked timespan',f'{selected.time_s.max()-selected.time_s.min():.1f} s')
            fig=go.Figure()
            fig.add_trace(go.Histogram2d(x=selected.court_x_m,y=selected.court_y_m,nbinsx=18,nbinsy=36,colorscale='YlGn',showscale=True, name='Court presence'))
            fig.add_shape(type='rect',x0=0,y0=0,x1=COURT_W,y1=COURT_L,line_color='white',line_width=3)
            fig.add_hline(y=COURT_L/2,line_color='white',line_width=3)
            fig.add_hline(y=COURT_L/2-2.13,line_color='white',line_width=1,dash='dash')
            fig.add_hline(y=COURT_L/2+2.13,line_color='white',line_width=1,dash='dash')
            fig.update_layout(title='Court coverage heatmap (detections, not ball landings)',height=570,
                xaxis=dict(range=[-.5,COURT_W+.5],title='Court width (m)',scaleanchor='y',scaleratio=1),
                yaxis=dict(range=[COURT_L+.5,-.5],title='Court length (m)'),margin=dict(t=50,b=40,l=40,r=15))
            st.plotly_chart(fig,use_container_width=True)
            st.download_button('Download AI player tracks (CSV)',tracks.to_csv(index=False).encode(),'player_tracks.csv','text/csv')
    st.divider()
    st.subheader('2 · Rally outcome review')
    st.write('Record the last playable ball landing zone and whether the selected player returned it. Out balls should be excluded from playable opportunities.')
    with st.form('rally',clear_on_submit=True):
        r1,r2,r3=st.columns(3)
        with r1: who=st.selectbox('Player',players)
        with r2: t=st.number_input('Video timestamp (seconds)',min_value=0.0,max_value=float(max(info['seconds'],0.1)),step=.5)
        with r3: result=st.selectbox('Outcome',['Missed return','Successful return','Out / not playable'])
        a,b=st.columns(2)
        with a: depth=st.selectbox('Ball end location — depth',['Front (kitchen)','Middle','Back (baseline)'])
        with b: side=st.selectbox('Ball end location — width',['Left','Centre','Right'])
        note=st.text_input('Optional note','')
        if st.form_submit_button('Add reviewed rally'):
            st.session_state.events.append({'Match':match_name,'Player':who,'Timestamp (s)':t,'Outcome':result,'Depth':depth,'Width':side,'Zone':depth+' / '+side,'Note':note})
            st.success('Rally saved in this session.')
    events=pd.DataFrame(st.session_state.events)
    if not events.empty:
        st.dataframe(events,use_container_width=True,hide_index=True)
        if st.button('Undo last rally'):
            st.session_state.events.pop();st.rerun()
        st.subheader('3 · Missed-return statistics')
        playable=events[events.Outcome.isin(['Missed return','Successful return'])].copy()
        if playable.empty: st.info('Record at least one playable ball for the percentage report.')
        else:
            playable['Missed']=(playable.Outcome=='Missed return').astype(int)
            stats=playable.groupby('Player').agg(Opportunities=('Missed','size'),Misses=('Missed','sum')).reset_index()
            stats['Miss %']=(100*stats.Misses/stats.Opportunities).round(1)
            st.dataframe(stats,use_container_width=True,hide_index=True)
            player=st.selectbox('Player for missed-ball location map',stats.Player.tolist())
            p=playable[playable.Player==player]
            order_depth=['Front (kitchen)','Middle','Back (baseline)'];order_width=['Left','Centre','Right']
            cells=p.groupby(['Depth','Width'])['Missed'].agg(['sum','count']).reset_index()
            z=[]; labels=[]
            for d in order_depth:
                zz=[]; ll=[]
                for wi in order_width:
                    row=cells[(cells.Depth==d)&(cells.Width==wi)]
                    if row.empty: zz.append(None);ll.append('No data')
                    else:
                        m=int(row.iloc[0]['sum']);n=int(row.iloc[0]['count'])
                        zz.append(round(100*m/n,1));ll.append(f'{m}/{n} missed')
                z.append(zz); labels.append(ll)
            heat=go.Figure(go.Heatmap(z=z,x=order_width,y=order_depth,text=labels,
                texttemplate='%{z:.0f}%<br>%{text}',colorscale='YlOrRd',zmin=0,zmax=100,hoverongaps=False,
                colorbar=dict(title='Miss %')))
            heat.update_layout(title=f'{player} — missed-return % by ball end zone',height=420,
                yaxis=dict(autorange='reversed'),margin=dict(t=55,b=30,l=50,r=20))
            st.plotly_chart(heat,use_container_width=True)
            st.caption('Missed-return % = missed returns ÷ (missed + successful playable returns) × 100. Blank = no reviewed balls. This is not AI-confirmed ball tracking.')
        st.download_button('Download rally review (CSV)',events.to_csv(index=False).encode(),'pickleball_rallies.csv','text/csv')
else:
    st.subheader('How it works')
    st.write('Upload a fixed-camera pickleball match video or load a public YouTube link to unlock player tracking and the review-based missed-return dashboard.')
    st.markdown('**Workflow:** Upload video or paste YouTube link → Calibrate 4 court corners → Run AI player tracking → Review rally outcomes and ball end zones → Compare missed-return percentages → Export CSV.')
    st.info('The application does not infer failed returns from a ball’s final position alone. This initial version uses manual outcome review for accurate denominators.')

st.caption('Prototype • Video processing occurs on the Python application server. Use an appropriately secured/private deployment for personal match footage.')
