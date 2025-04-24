---
layout: page
title: "Audio Samples Demo"
date:   2024-06-16 17:50:00
---
# AdvSpeech V2
the newest solution, it's white box(for now) and based on the gradient, we benchmarked it on Cosyvoice and Spark-TTS.
### Spark-TTS
<table>

  <tr>
    <th></th>
    <th>Speech Prompt</th>
    <th>Synthetic</th>
  </tr>

  <tr>
    <th></th>
    <th>As the doctors entered the street, they saw a man in a cassock standing on the threshold of the next door</th>
    <th>A quick brown fox jumps over the lazy dog</th>
  </tr>

  <tr>
    <th>Unprotected</th>
    <td>
      <audio controls>
        <source src="/audios/en_sample/libri_5694.wav" type="audio/wav">
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="/audios/en_sample/libri_cosyvoice.wav" type="audio/wav">
      </audio>
    </td>
  </tr>


  <tr>
    <th>Ours</th>
    <td>
      <audio controls>
        <source src="/audios/V2_Spark/adv424200.wav" type="audio/wav">
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="/audios/V2_Spark/adv424200_syn_spark.wav" type="audio/wav">
      </audio>
    </td>
  </tr>
</table>

# AdvSpeech V1
v1 is a black box solution based on speech envelope. The goal is to shadow the original speaker's voice and style during ZS-TTS
### EN SAMPLE

<table>

  <tr>
    <th></th>
    <th>Speaker Utterance</th>
    <th>Synthesis Result</th>
  </tr>

  <tr>
    <th></th>
    <th>I do not know what it was, but I heard Gladden say, "Tell General Bragg that I have as keen a scent for Yankees as General Chalmers has."</th>
    <th>She was a beautiful girl of about seventeen years of age, not fat like all the rest of the Pinkies, but slender and well formed according to our own ideas of beauty.</th>
  </tr>

  <tr>
    <th>Original</th>
    <td>
      <audio controls>
        <source src="/audios/en_sample/libri_5694.wav" type="audio/wav">
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="/audios/en_sample/libri_cosyvoice.wav" type="audio/wav">
      </audio>
    </td>
  </tr>


  <tr>
    <th>Antifake</th>
    <td>
      <audio controls>
        <source src="/audios/en_sample/libri_antifake.wav" type="audio/wav">
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="/audios/en_sample/libri_antifake_cosyvoice.wav" type="audio/wav">
      </audio>
    </td>
  </tr>


  <tr>
    <th>Ours</th>
    <td>
      <audio controls>
        <source src="/audios/en_sample/libri_adv.wav" type="audio/wav">
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="/audios/en_sample/libri_adv_cosyvoice.wav" type="audio/wav">
      </audio>
    </td>
  </tr>
</table>

### CN SAMPLE

<table>

  <tr>
    <th></th>
    <th>Speaker Utterance</th>
    <th>Synthesis Result</th>
  </tr>

  <tr>
    <th></th>
    <th>欲问后期何日是, 寄书应见雁南征</th>
    <th>今天阳光明媚，我去公园散步，看见很多人放风筝。</th>
  </tr>

  <tr>
    <th>Original</th>
    <td>
      <audio controls>
        <source src="/audios/cn_sample/original.wav" type="audio/wav">
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="/audios/cn_sample/original_cosyvoice.wav" type="audio/wav">
      </audio>
    </td>
  </tr>


  <tr>
    <th>Antifake</th>
    <td>
      <audio controls>
        <source src="/audios/cn_sample/antifake.wav" type="audio/wav">
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="/audios/cn_sample/antifake_cosyvoice.wav" type="audio/wav">
      </audio>
    </td>
  </tr>


  <tr>
    <th>Ours</th>
    <td>
      <audio controls>
        <source src="/audios/cn_sample/ry_adv.wav" type="audio/wav">
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="/audios/cn_sample/ours_sfm_ws_ref_cosyvoice.wav" type="audio/wav">
      </audio>
    </td>
  </tr>
</table>
