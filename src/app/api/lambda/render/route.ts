import {
  AwsRegion,
  renderMediaOnLambda,
  RenderMediaOnLambdaOutput,
  speculateFunctionName,
} from "@remotion/lambda/client";
import {
  DISK,
  RAM,
  REGION,
  SITE_NAME,
  TIMEOUT,
} from "../../../../../config.mjs";
import { RenderRequest } from "../../../../../types/schema";
import { executeApi } from "../../../../helpers/api-response";

const validateEnvVars = () => {
  const missing: string[] = [];
  if (
    !process.env.AWS_ACCESS_KEY_ID &&
    !process.env.REMOTION_AWS_ACCESS_KEY_ID
  ) {
    missing.push("AWS_ACCESS_KEY_ID (or REMOTION_AWS_ACCESS_KEY_ID)");
  }
  if (
    !process.env.AWS_SECRET_ACCESS_KEY &&
    !process.env.REMOTION_AWS_SECRET_ACCESS_KEY
  ) {
    missing.push(
      "AWS_SECRET_ACCESS_KEY (or REMOTION_AWS_SECRET_ACCESS_KEY)",
    );
  }
  if (missing.length > 0) {
    throw new Error(
      `Missing environment variables: ${missing.join(", ")}. Set up Remotion Lambda — see the README.md.`,
    );
  }
};

export const POST = executeApi<RenderMediaOnLambdaOutput, typeof RenderRequest>(
  RenderRequest,
  async (req, body) => {
    validateEnvVars();
    const result = await renderMediaOnLambda({
      codec: "h264",
      functionName: speculateFunctionName({
        diskSizeInMb: DISK,
        memorySizeInMb: RAM,
        timeoutInSeconds: TIMEOUT,
      }),
      region: REGION as AwsRegion,
      serveUrl: SITE_NAME,
      composition: body.id,
      inputProps: body.inputProps,
      framesPerLambda: 10,
      downloadBehavior: {
        type: "download",
        fileName: "video.mp4",
      },
    });

    return result;
  },
);
