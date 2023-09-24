import React from 'react';

export default function Footer() {
  return (
    <div className="hidden px-3 pb-1 pt-2 text-center text-xs text-black/50 dark:text-white/50 md:block md:px-4 md:pb-4 md:pt-3">
      ALuvU: AI for Learning, Understanding, and Valuing You. ChatGPT 사용 팁이나 의견 있으면
      슬랙에서 같이 이야기 나눠요. 😊{' '}
      <a
        href="https://join.slack.com/t/aluvuhq/shared_invite/zt-20zzxpmcy-DfVmH58D1m21pyiwPGNlOQ"
        target="_blank"
        rel="noreferrer"
        className="underline"
      >
        Join Slack
      </a>
    </div>
  );
}
