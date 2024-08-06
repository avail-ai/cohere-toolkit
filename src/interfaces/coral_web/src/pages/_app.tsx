import {
  DehydratedState,
  QueryClient,
  QueryClientProvider,
} from '@tanstack/react-query';
// import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import fetch from 'cross-fetch';
import type { AppProps } from 'next/app';

import { CohereClient, CohereClientProvider, Fetch } from '@/cohere-client';
import { ToastNotification } from '@/components/Shared';
import { WebManifestHead } from '@/components/Shared';
import { GlobalHead } from '@/components/Shared/GlobalHead';
import { ViewportFix } from '@/components/ViewportFix';
import { ContextStore } from '@/context';
import { env } from '@/env.mjs';
import { useLazyRef } from '@/hooks/lazyRef';
import '@/styles/main.css';
import { PageProvider } from '@/page-context';

/**
 * Create a CohereAPIClient with the given access token.
 */
const makeCohereClient = (authToken?: string) => {
  const apiFetch: Fetch = async (resource, config) => await fetch(resource, config);
  return new CohereClient({
    hostname: env.NEXT_PUBLIC_API_HOSTNAME,
    fetch: apiFetch,
    authToken,
  });
};




const App: React.FC<AppProps> = ({ Component, pageProps, ...props }) => {
  const cohereClient = useLazyRef(() => makeCohereClient());
  const queryClient = useLazyRef(() => new QueryClient());

  return (
    <CohereClientProvider client={cohereClient}>
      <QueryClientProvider client={queryClient}>
        <ContextStore>
          <PageProvider>
            <ViewportFix />
            <GlobalHead />
            <WebManifestHead />
            <ToastNotification />
            {/* <ReactQueryDevtools /> */}
            <Component {...pageProps} />
          </PageProvider>
        </ContextStore>
      </QueryClientProvider>
    </CohereClientProvider>
  );
};

export default App;